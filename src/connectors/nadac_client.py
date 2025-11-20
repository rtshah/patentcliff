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
        Query Medicaid DKAN catalog to discover year -> UUID mapping for NADAC datasets.
        Returns and caches the registry.
        """
        # DKAN catalog endpoint
        catalog_url = f"{self.base_url}/datastore/1/metastore/schemas/dataset/items"
        
        headers = {}
        if self.app_token:
            headers["X-App-Token"] = self.app_token
        
        try:
            response = requests.get(catalog_url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            datasets = response.json()
            
            # Filter for NADAC datasets
            nadac_datasets = {}
            for dataset in datasets:
                title = dataset.get("title", "").upper()
                if "NADAC" in title:
                    # Try to extract year from title
                    import re
                    year_match = re.search(r'20\d{2}', title)
                    if year_match:
                        year = year_match.group()
                        uuid = dataset.get("identifier") or dataset.get("id")
                        if uuid:
                            nadac_datasets[year] = uuid
            
            self.uuid_registry.update(nadac_datasets)
            self._save_registry()
            
            print(f"Discovered {len(nadac_datasets)} NADAC datasets")
            return self.uuid_registry
        
        except Exception as e:
            print(f"Error discovering NADAC UUID registry: {e}")
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
    
    def fetch_nadac_month(
        self, 
        ndcs: List[str], 
        year_month: str
    ) -> pd.DataFrame:
        """
        Fetch NADAC data for given NDCs in a specific month.
        
        Args:
            ndcs: List of 11-digit package_ndc strings
            year_month: Format "YYYY-MM"
        
        Returns:
            DataFrame with columns: ndc, effective_date, nadac_per_unit, pricing_unit
        """
        year = year_month.split("-")[0]
        uuid = self._get_dataset_uuid(year)
        
        if not uuid:
            print(f"No UUID found for year {year}, attempting discovery...")
            self.discover_uuid_registry()
            uuid = self._get_dataset_uuid(year)
            if not uuid:
                print(f"Warning: Could not find NADAC dataset for {year}")
                return pd.DataFrame(columns=["ndc", "effective_date", "nadac_per_unit", "pricing_unit"])
        
        # Build query for DKAN datastore
        # Format: base_url/datastore/1/metastore/schemas/dataset/items/{uuid}/data
        data_url = f"{self.base_url}/datastore/1/metastore/schemas/dataset/items/{uuid}/data"
        
        headers = {}
        if self.app_token:
            headers["X-App-Token"] = self.app_token
        
        # Query parameters
        # Note: DKAN API may require specific query format
        # For now, fetch all and filter in memory (may need optimization)
        params = {
            "limit": 10000,  # Adjust based on API limits
        }
        
        try:
            response = requests.get(data_url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse response (may be JSON or CSV)
            data = response.json()
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and "results" in data:
                df = pd.DataFrame(data["results"])
            else:
                df = pd.DataFrame()
            
            if df.empty:
                return pd.DataFrame(columns=["ndc", "effective_date", "nadac_per_unit", "pricing_unit"])
            
            # Normalize column names
            column_mapping = {
                "ndc": ["ndc", "NDC", "package_ndc", "PACKAGE_NDC"],
                "effective_date": ["effective_date", "EFFECTIVE_DATE", "as_of_date"],
                "nadac_per_unit": ["nadac_per_unit", "NADAC_PER_UNIT", "nadac", "NADAC"],
                "pricing_unit": ["pricing_unit", "PRICING_UNIT", "unit", "UNIT"],
            }
            
            for std_name, aliases in column_mapping.items():
                for alias in aliases:
                    if alias in df.columns:
                        if std_name not in df.columns or df.columns.get_loc(alias) < df.columns.get_loc(std_name):
                            df = df.rename(columns={alias: std_name})
                        break
            
            # Filter to requested NDCs
            if "ndc" in df.columns:
                # Normalize NDC format (remove dashes, pad)
                df["ndc_normalized"] = df["ndc"].astype(str).str.replace("-", "").str.replace(" ", "")
                df["ndc_normalized"] = df["ndc_normalized"].apply(
                    lambda x: "0" + x if len(x) == 10 else x if len(x) == 11 else ""
                )
                
                ndcs_normalized = [n.replace("-", "").replace(" ", "") for n in ndcs]
                ndcs_normalized = ["0" + n if len(n) == 10 else n for n in ndcs_normalized]
                
                df = df[df["ndc_normalized"].isin(ndcs_normalized)]
            
            # Filter to target month
            if "effective_date" in df.columns:
                df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
                df = df[df["effective_date"].notna()]
                df["year_month"] = df["effective_date"].dt.to_period("M").astype(str)
                df = df[df["year_month"] == year_month]
            
            # Select and return relevant columns
            result_cols = ["ndc", "effective_date", "nadac_per_unit", "pricing_unit"]
            available_cols = [c for c in result_cols if c in df.columns]
            
            return df[available_cols].copy() if available_cols else pd.DataFrame(columns=result_cols)
        
        except Exception as e:
            print(f"Error fetching NADAC data for {year_month}: {e}")
            return pd.DataFrame(columns=["ndc", "effective_date", "nadac_per_unit", "pricing_unit"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--discover-registry", action="store_true")
    args = parser.parse_args()
    
    client = NADACClient()
    if args.discover_registry:
        registry = client.discover_uuid_registry()
        print(f"Registry: {json.dumps(registry, indent=2)}")

