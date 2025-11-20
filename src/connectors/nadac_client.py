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
    
    def discover_uuid_registry(self) -> Dict[str, str]:
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
            
            print(f"Discovered {len(nadac_datasets)} NADAC datasets")
            if nadac_datasets:
                print(f"Years found: {sorted(nadac_datasets.keys())}")
            return self.uuid_registry
        
        except Exception as e:
            print(f"Error discovering NADAC UUID registry: {e}")
            import traceback
            traceback.print_exc()
            # Return existing registry if discovery fails
            return self.uuid_registry
    
    def _get_year_from_date(self, target_date: date) -> str:
        """Get year string from date."""
        return str(target_date.year)
    
    def _get_dataset_uuid(self, year: str) -> Optional[str]:
        """Get dataset UUID for a given year."""
        if year in self.uuid_registry:
            return self.uuid_registry[year]
        
        # Try discovery if not found
        self.discover_uuid_registry()
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
        page_size: int = 5000
    ) -> pd.DataFrame:
        """
        Fetch NADAC data for given NDCs in a specific month.
        Uses server-side filtering for efficiency.
        
        Args:
            ndcs: List of 11-digit package_ndc strings
            year_month: Format "YYYY-MM"
            page_size: Number of rows per page (default 5000)
        
        Returns:
            DataFrame with columns: ndc, effective_date, nadac_per_unit, pricing_unit
        """
        def norm_ndc(s: str) -> str:
            """Normalize NDC to 11 digits."""
            x = (s or "").replace("-", "").replace(" ", "").strip()
            return ("0" + x) if len(x) == 10 else x
        
        ndcs_norm = [norm_ndc(n) for n in ndcs if n]
        if not ndcs_norm:
            return pd.DataFrame(columns=["ndc", "effective_date", "nadac_per_unit", "pricing_unit"])
        
        year = year_month.split("-")[0]
        rid = self._get_dataset_uuid(year) or self.discover_uuid_registry().get(year)
        
        if not rid:
            print(f"Warning: Could not find NADAC dataset for {year}")
            return pd.DataFrame(columns=["ndc", "effective_date", "nadac_per_unit", "pricing_unit"])
        
        # Build date range for server-side filtering
        from calendar import monthrange
        y, m = map(int, year_month.split("-"))
        last_day = monthrange(y, m)[1]
        between_val = f"{year_month}-01/{year_month}-{last_day:02d}"
        
        frames = []
        CHUNK = 100  # Use 'in' operator which can handle more values
        
        # Process NDCs in chunks
        for i in range(0, len(ndcs_norm), CHUNK):
            ndc_chunk = ndcs_norm[i:i+CHUNK]
            
            # Build conditions: date range + NDC filter
            # Use >= and <= for dates (more reliable than 'between')
            conditions = [
                {"property": "effective_date", "operator": ">=", "value": f"{year_month}-01"},
                {"property": "effective_date", "operator": "<=", "value": f"{year_month}-{last_day:02d}"}
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
        
        # Normalize effective_date and filter to exact month (client-side check)
        if "effective_date" in df.columns:
            df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
            df = df[df["effective_date"].notna()]
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
        
        return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--discover-registry", action="store_true")
    args = parser.parse_args()
    
    client = NADACClient()
    if args.discover_registry:
        registry = client.discover_uuid_registry()
        print(f"Registry: {json.dumps(registry, indent=2)}")

