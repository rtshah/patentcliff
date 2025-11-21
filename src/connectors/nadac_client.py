"""NADAC (National Average Drug Acquisition Cost) API client."""
import requests
import os
import json
from pathlib import Path
from typing import List, Optional, Dict
from datetime import date
import pandas as pd
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config


class NADACClient:
    """Client for Medicaid DKAN NADAC API."""
    
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = load_config()
        self.config = config
        self.base_url = config["apis"]["medicaid_dkan"]["base_url"]
        self.timeout = config["apis"]["medicaid_dkan"]["timeout_s"]
        self.app_token = os.getenv(config["apis"]["medicaid_dkan"]["app_token_env"])
        self.cache_dir = Path(config["paths"]["api_cache_dir"]) / "nadac"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = Path(config["paths"]["cache_dir"]) / "nadac_uuid_registry.json"
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.uuid_registry = self._load_registry()
        
        # Performance optimization: track discovery state and year bounds
        self._warned_years = set()
        self._discovered_once = False
        self.min_year = None
        self.max_year = None
        self._month_cache = {}  # Optional memoization per month
        
        # Discover once if registry is empty
        if not self.uuid_registry:
            self.discover_uuid_registry(silent=True)
        else:
            self._update_year_bounds()
    
    def _load_registry(self) -> Dict[str, str]:
        """Load UUID registry from cache or config override."""
        # Check config override first
        config_override = self.config["apis"]["medicaid_dkan"].get("nadac_uuid_registry", {})
        if config_override:
            return config_override
        
        # Load from cache
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        
        return {}
    
    def _save_registry(self):
        """Save UUID registry to cache."""
        with open(self.registry_path, "w") as f:
            json.dump(self.uuid_registry, f, indent=2)
    
    def _update_year_bounds(self):
        """Update min_year and max_year from registry."""
        if not self.uuid_registry:
            self.min_year = None
            self.max_year = None
            return
        
        years = sorted(int(y) for y in self.uuid_registry.keys() if y.isdigit())
        self.min_year = min(years) if years else None
        self.max_year = max(years) if years else None
    
    def discover_uuid_registry(self, silent: bool = False) -> Dict[str, str]:
        """
        Query Medicaid DKAN catalog to discover year -> datastore UUID mapping for NADAC datasets.
        Returns and caches the registry.
        
        Uses the search API to find NADAC datasets, then extracts datastore identifiers
        from the distribution references.
        """
        # Use search API to find NADAC datasets
        search_url = f"{self.base_url}/search"
        
        headers = {}
        if self.app_token:
            headers["X-App-Token"] = self.app_token
        
        params = {
            "fulltext": "NADAC",
            "page-size": 100,  # Get as many as possible
        }
        
        try:
            response = requests.get(search_url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            search_results = response.json()
            
            # Filter for NADAC datasets
            # Store both dataset_id (for query endpoint) and resource_id (from ref)
            nadac_datasets = {}
            results = search_results.get("results", {})
            
            for dataset_key, dataset in results.items():
                title = dataset.get("title", "").upper()
                if "NADAC" in title and "NATIONAL AVERAGE DRUG ACQUISITION COST" in title:
                    # Try to extract year from title
                    import re
                    year_match = re.search(r'20\d{2}', title)
                    if year_match:
                        year = year_match.group()
                        dataset_id = dataset.get("identifier")
                        
                        # Get datastore resource identifier from distribution
                        distributions = dataset.get("distribution", [])
                        for dist_idx, dist in enumerate(distributions):
                            # Check for datastore reference in %Ref:downloadURL
                            ref_download = dist.get("%Ref:downloadURL", [])
                            if ref_download and len(ref_download) > 0:
                                resource_id = ref_download[0].get("identifier")
                                if resource_id and dataset_id:
                                    # Store as: dataset_id|resource_id|distribution_index
                                    # This allows us to try different query formats
                                    nadac_datasets[year] = f"{dataset_id}|{resource_id}|{dist_idx}"
                                    break
                            
                            # Fallback: use dataset identifier with distribution index
                            if dataset_id:
                                nadac_datasets[year] = f"{dataset_id}||{dist_idx}"
                                break
            
            self.uuid_registry.update(nadac_datasets)
            self._save_registry()
            
            if not silent:
                print(f"Discovered {len(nadac_datasets)} NADAC datasets")
                if nadac_datasets:
                    print(f"Years found: {sorted(nadac_datasets.keys())}")
            
            self._discovered_once = True
            self._update_year_bounds()
            return self.uuid_registry
        
        except Exception as e:
            if not silent:
                print(f"Error discovering NADAC UUID registry: {e}")
                import traceback
                traceback.print_exc()
            # Mark as discovered (even if failed) to prevent repeated attempts
            self._discovered_once = True
            self._update_year_bounds()
            # Return existing registry if discovery fails
            return self.uuid_registry
    
    def _get_year_from_date(self, target_date: date) -> str:
        """Get year string from date."""
        return str(target_date.year)
    
    def _get_dataset_uuid(self, year: str) -> Optional[str]:
        """
        Get dataset UUID for a given year.
        
        Do NOT call discover here; just return what we have.
        Discovery should happen once at init or explicitly.
        """
        return self.uuid_registry.get(year)
    
    def _query_dkan_datastore(self, resource_info: str, body: dict) -> dict:
        """
        Query DKAN datastore with proper endpoint format and retry/backoff.
        
        Args:
            resource_info: Format "dataset_id|resource_id|dist_idx" or "dataset_id||dist_idx"
            body: Query payload
        
        Returns:
            JSON response from API
        """
        # Parse resource info
        parts = resource_info.split("|")
        dataset_id = parts[0]
        resource_id = parts[1] if len(parts) > 1 and parts[1] else None
        dist_idx = parts[2] if len(parts) > 2 else "0"
        
        # Try dataset_id format first (most reliable)
        url = f"{self.base_url}/datastore/query/{dataset_id}/{dist_idx}"
        
        headers = {"Content-Type": "application/json"}
        if self.app_token:
            headers["X-App-Token"] = self.app_token
        
        # Retry with backoff for 503 errors
        for attempt in range(3):
            try:
                r = requests.post(url, json=body, headers=headers, timeout=self.timeout)
                if r.status_code == 503 and attempt < 2:
                    import time
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if r.status_code == 404 and resource_id and attempt == 0:
                    # Try resource_id format as fallback
                    url = f"{self.base_url}/datastore/query/{resource_id}/0"
                    continue
                r.raise_for_status()
                return r.json()
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    raise
                import time
                time.sleep(1.5 * (attempt + 1))
        
        return {}
    
    def fetch_nadac_month(
        self, 
        ndcs: List[str], 
        year_month: str,
        page_size: int = 5000,
        snap_days: int = 14
    ) -> pd.DataFrame:
        """
        Fetch NADAC data for given NDCs in a specific month.
        Uses server-side filtering for efficiency.
        Supports date snapping to nearest published date within ±snap_days.
        
        Args:
            ndcs: List of 11-digit package_ndc strings
            year_month: Format "YYYY-MM" (target month)
            page_size: Number of rows per page (default 5000)
            snap_days: Days to expand search window for date snapping (default 14)
        
        Returns:
            DataFrame with columns: ndc, effective_date, nadac_per_unit, pricing_unit
            Results are snapped to nearest published date within ±snap_days
        """
        def norm_ndc(s: str) -> str:
            """Normalize NDC to 11 digits."""
            x = (s or "").replace("-", "").replace(" ", "").strip()
            return ("0" + x) if len(x) == 10 else x
        
        ndcs_norm = [norm_ndc(n) for n in ndcs if n]
        if not ndcs_norm:
            return pd.DataFrame(columns=["ndc", "effective_date", "nadac_per_unit", "pricing_unit"])
        
        # Optional memoization: check cache first
        key = (tuple(sorted(set(ndcs_norm))), year_month)
        if key in self._month_cache:
            return self._month_cache[key].copy()
        
        year = year_month.split("-")[0]
        
        # Short-circuit out-of-range years (and warn once)
        if self.min_year and self.max_year:
            try:
                year_int = int(year)
                if year_int < self.min_year or year_int > self.max_year:
                    if year not in self._warned_years:
                        print(f"NADAC: skipping {year} (outside [{self.min_year}, {self.max_year}])")
                        self._warned_years.add(year)
                    # Return consistent empty frame
                    empty_df = pd.DataFrame(columns=["ndc", "effective_date", "nadac_per_unit", "pricing_unit"])
                    self._month_cache[key] = empty_df.copy()
                    return empty_df
            except ValueError:
                # Invalid year format
                pass
        
        # Get dataset UUID (do NOT discover here)
        rid = self._get_dataset_uuid(year)
        
        # One last attempt: refresh once per run if not discovered yet
        if not rid and not self._discovered_once:
            self.discover_uuid_registry(silent=True)
            rid = self._get_dataset_uuid(year)
        
        if not rid:
            if year not in self._warned_years:
                print(f"NADAC: no dataset for {year}")
                self._warned_years.add(year)
            empty_df = pd.DataFrame(columns=["ndc", "effective_date", "nadac_per_unit", "pricing_unit"])
            self._month_cache[key] = empty_df.copy()
            return empty_df
        
        # Build date range for server-side filtering
        # Expand by ±snap_days to allow date snapping
        from calendar import monthrange
        from datetime import datetime, timedelta
        y, m = map(int, year_month.split("-"))
        last_day = monthrange(y, m)[1]
        
        # Target date is middle of month (or first day if snap_days=0)
        target_date = datetime(y, m, 1)
        if snap_days > 0:
            # Use middle of month as target for snapping
            target_date = datetime(y, m, min(15, last_day))
        
        # Expand date range by ±snap_days
        date_start = (target_date - timedelta(days=snap_days)).strftime("%Y-%m-%d")
        date_end = (target_date + timedelta(days=snap_days)).strftime("%Y-%m-%d")
        
        # Ensure we don't go outside the month boundaries (for efficiency)
        month_start = f"{year_month}-01"
        month_end = f"{year_month}-{last_day:02d}"
        
        frames = []
        CHUNK = 100  # Use 'in' operator which can handle more values
        
        # Process NDCs in chunks
        for i in range(0, len(ndcs_norm), CHUNK):
            ndc_chunk = ndcs_norm[i:i+CHUNK]
            
            # Build conditions: date range + NDC filter
            # Use expanded date range for snapping
            conditions = [
                {"property": "effective_date", "operator": ">=", "value": date_start},
                {"property": "effective_date", "operator": "<=", "value": date_end}
            ]
            
            # Add NDC filter using 'in' operator
            if len(ndc_chunk) == 1:
                conditions.append({"property": "ndc", "operator": "=", "value": ndc_chunk[0]})
            else:
                conditions.append({"property": "ndc", "operator": "in", "value": ndc_chunk})
            
            body = {
                "limit": page_size,
                "offset": 0,
                "conditions": conditions,
                "sorts": [{"property": "effective_date", "order": "asc"}],
            }
            
            # Paginate through results
            while True:
                try:
                    resp = self._query_dkan_datastore(rid, body)
                    rows = resp.get("results") or []
                    
                    if not rows:
                        break
                    
                    frames.append(pd.DataFrame(rows))
                    
                    if len(rows) < page_size:
                        break
                    
                    body["offset"] += page_size
                except Exception as e:
                    print(f"Error fetching chunk {i//CHUNK + 1}: {e}")
                    break
        
        if not frames:
            return pd.DataFrame(columns=["ndc", "effective_date", "nadac_per_unit", "pricing_unit"])
        
        df = pd.concat(frames, ignore_index=True)
        
        # Normalize column names
        rename = {
            "NDC": "ndc",
            "PACKAGE_NDC": "ndc",
            "EFFECTIVE_DATE": "effective_date",
            "NADAC_PER_UNIT": "nadac_per_unit",
            "PRICING_UNIT": "pricing_unit"
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns and v not in df.columns})
        
        # Keep only needed columns
        keep = [c for c in ["ndc", "effective_date", "nadac_per_unit", "pricing_unit"] if c in df.columns]
        df = df[keep].copy()
        
        # Normalize effective_date and apply date snapping
        if "effective_date" in df.columns:
            df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
            df = df[df["effective_date"].notna()]
            
            if snap_days > 0 and len(df) > 0:
                # Date snapping: for each NDC, keep the row with date closest to target
                target_dt = pd.Timestamp(target_date)
                df["_date_diff"] = (df["effective_date"] - target_dt).abs()
                
                # Group by NDC and keep row with minimum date difference
                df = df.loc[df.groupby("ndc")["_date_diff"].idxmin()].copy()
                df = df.drop(columns=["_date_diff"])
                
                # Filter to ensure we're within snap window
                df = df[
                    (df["effective_date"] >= pd.Timestamp(date_start)) &
                    (df["effective_date"] <= pd.Timestamp(date_end))
                ]
            else:
                # No snapping: filter to exact month
                df = df[df["effective_date"].dt.to_period("M").astype(str) == year_month]
        
        # Clean nadac_per_unit (remove $ and commas)
        if "nadac_per_unit" in df.columns:
            df["nadac_per_unit"] = (
                df["nadac_per_unit"].astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            df["nadac_per_unit"] = pd.to_numeric(df["nadac_per_unit"], errors="coerce")
        
        # Normalize NDC format
        if "ndc" in df.columns:
            df["ndc"] = (
                df["ndc"].astype(str)
                .str.replace("-", "")
                .str.strip()
                .apply(lambda x: "0" + x if len(x) == 10 else x)
            )
        
        # Cache result before returning
        self._month_cache[key] = df.copy()
        return df
    
    def fetch_nadac_with_fallback(
        self,
        ndcs: List[str],
        year_month: str,
        page_size: int = 5000,
        snap_days: int = 14
    ):
        """
        Fetch NADAC data with NDC-9 fallback.
        
        First tries exact NDC-11 match. If no results, falls back to NDC-9
        (labeler+product) and takes median price across packages.
        
        Args:
            ndcs: List of 11-digit package_ndc strings
            year_month: Format "YYYY-MM"
            page_size: Number of rows per page
            snap_days: Days to expand search window
        
        Returns:
            Tuple of (DataFrame with prices, Series of join flags)
            Join flags: True if joined on NDC-11, False if joined on NDC-9
        """
        def norm_ndc(s: str) -> str:
            """Normalize NDC to 11 digits."""
            x = (s or "").replace("-", "").replace(" ", "").strip()
            return ("0" + x) if len(x) == 10 else x
        
        ndcs_norm = [norm_ndc(n) for n in ndcs if n]
        if not ndcs_norm:
            empty_df = pd.DataFrame(columns=["ndc", "effective_date", "nadac_per_unit", "pricing_unit", "joined_on_ndc11"])
            empty_flags = pd.Series([False] * len(ndcs), index=ndcs, name="joined_on_ndc11")
            return empty_df, empty_flags
        
        # Try NDC-11 exact match first
        result_11 = self.fetch_nadac_month(ndcs_norm, year_month, page_size, snap_days)
        
        # Track which NDCs matched on NDC-11
        matched_11 = set(result_11["ndc"].unique()) if not result_11.empty else set()
        join_flags = pd.Series(
            [ndc in matched_11 for ndc in ndcs_norm],
            index=ndcs_norm,
            name="joined_on_ndc11"
        )
        
        # For NDCs that didn't match on NDC-11, try NDC-9 fallback
        unmatched_ndcs = [ndc for ndc in ndcs_norm if ndc not in matched_11]
        
        if not unmatched_ndcs:
            # All matched on NDC-11
            result_11["joined_on_ndc11"] = True
            return result_11, join_flags
        
        # Extract NDC-9 (first 9 digits = labeler+product)
        ndc9_to_ndc11 = {}
        for ndc11 in unmatched_ndcs:
            ndc9 = ndc11[:9]
            if ndc9 not in ndc9_to_ndc11:
                ndc9_to_ndc11[ndc9] = []
            ndc9_to_ndc11[ndc9].append(ndc11)
        
        # Fetch all NADAC data for the month (we'll filter client-side)
        # Use a broader query to get all NDCs starting with our NDC-9 prefixes
        year = year_month.split("-")[0]
        rid = self._get_dataset_uuid(year)
        if not rid:
            result_11["joined_on_ndc11"] = True
            return result_11, join_flags
        
        # Build date range
        from calendar import monthrange
        from datetime import datetime, timedelta
        y, m = map(int, year_month.split("-"))
        target_date = datetime(y, m, min(15, monthrange(y, m)[1]))
        date_start = (target_date - timedelta(days=snap_days)).strftime("%Y-%m-%d")
        date_end = (target_date + timedelta(days=snap_days)).strftime("%Y-%m-%d")
        
        # Query for NDCs matching our NDC-9 prefixes
        # We'll use a LIKE pattern if available, otherwise fetch and filter
        frames_9 = []
        for ndc9 in ndc9_to_ndc11.keys():
            # Query for NDCs starting with this prefix
            # NADAC API doesn't support prefix matching, so we fetch and filter
            # For efficiency, query a reasonable range
            body = {
                "limit": page_size,
                "offset": 0,
                "conditions": [
                    {"property": "effective_date", "operator": ">=", "value": date_start},
                    {"property": "effective_date", "operator": "<=", "value": date_end}
                ],
                "sorts": [{"property": "effective_date", "order": "asc"}]
            }
            
            # Fetch a batch and filter client-side
            try:
                resp = self._query_dkan_datastore(rid, body)
                results = resp.get("results", [])
                
                # Filter to NDCs starting with our prefix
                matching = [
                    r for r in results
                    if r.get("ndc", "").replace("-", "").replace(" ", "").startswith(ndc9)
                ]
                
                if matching:
                    df_chunk = pd.DataFrame(matching)
                    frames_9.append(df_chunk)
            except Exception as e:
                continue
        
        if not frames_9:
            # No NDC-9 matches either
            result_11["joined_on_ndc11"] = True
            return result_11, join_flags
        
        # Combine NDC-9 results
        df_9 = pd.concat(frames_9, ignore_index=True) if frames_9 else pd.DataFrame()
        
        # Normalize columns
        rename = {
            "NDC": "ndc",
            "PACKAGE_NDC": "ndc",
            "EFFECTIVE_DATE": "effective_date",
            "NADAC_PER_UNIT": "nadac_per_unit",
            "Pricing_Unit": "pricing_unit"
        }
        df_9 = df_9.rename(columns={k: v for k, v in rename.items() if k in df_9.columns and v not in df_9.columns})
        
        # Normalize NDC format
        if "ndc" in df_9.columns:
            df_9["ndc"] = (
                df_9["ndc"].astype(str)
                .str.replace("-", "")
                .str.strip()
                .apply(lambda x: "0" + x if len(x) == 10 else x)
            )
            df_9["ndc9"] = df_9["ndc"].str[:9]
        
        # Normalize effective_date and apply date snapping
        if "effective_date" in df_9.columns:
            df_9["effective_date"] = pd.to_datetime(df_9["effective_date"], errors="coerce")
            df_9 = df_9[df_9["effective_date"].notna()]
            
            if snap_days > 0 and len(df_9) > 0:
                target_dt = pd.Timestamp(target_date)
                df_9["_date_diff"] = (df_9["effective_date"] - target_dt).abs()
                df_9 = df_9.loc[df_9.groupby("ndc9")["_date_diff"].idxmin()].copy()
                df_9 = df_9.drop(columns=["_date_diff"])
                df_9 = df_9[
                    (df_9["effective_date"] >= pd.Timestamp(date_start)) &
                    (df_9["effective_date"] <= pd.Timestamp(date_end))
                ]
        
        # Clean nadac_per_unit
        if "nadac_per_unit" in df_9.columns:
            df_9["nadac_per_unit"] = (
                df_9["nadac_per_unit"].astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            df_9["nadac_per_unit"] = pd.to_numeric(df_9["nadac_per_unit"], errors="coerce")
        
        # Group by NDC-9 and take median price per pricing_unit
        # Map back to original NDC-11s
        result_9_rows = []
        for ndc9, ndc11_list in ndc9_to_ndc11.items():
            ndc9_data = df_9[df_9["ndc9"] == ndc9].copy()
            if ndc9_data.empty:
                continue
            
            # Group by pricing_unit and take median
            for unit in ndc9_data["pricing_unit"].dropna().unique():
                unit_data = ndc9_data[ndc9_data["pricing_unit"] == unit]
                median_price = unit_data["nadac_per_unit"].median()
                median_date = unit_data["effective_date"].median()
                
                # Create a row for each original NDC-11
                for ndc11 in ndc11_list:
                    result_9_rows.append({
                        "ndc": ndc11,
                        "effective_date": median_date,
                        "nadac_per_unit": median_price,
                        "pricing_unit": unit,
                        "joined_on_ndc11": False
                    })
        
        if result_9_rows:
            df_9_result = pd.DataFrame(result_9_rows)
            # Combine with NDC-11 results
            if not result_11.empty:
                result_11["joined_on_ndc11"] = True
                final_df = pd.concat([result_11, df_9_result], ignore_index=True)
            else:
                final_df = df_9_result
            
            # Update join flags
            for ndc11 in df_9_result["ndc"].unique():
                if ndc11 in join_flags.index:
                    join_flags[ndc11] = False
            
            return final_df, join_flags
        
        # No NDC-9 matches
        result_11["joined_on_ndc11"] = True
        return result_11, join_flags


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--discover-registry", action="store_true")
    args = parser.parse_args()
    
    client = NADACClient()
    if args.discover_registry:
        registry = client.discover_uuid_registry()
        print(f"Registry: {json.dumps(registry, indent=2)}")

