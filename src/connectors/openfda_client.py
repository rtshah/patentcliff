"""openFDA NDC API client for listing brand and generic NDCs."""
import requests
import os
import time
from typing import List, Set, Tuple, Optional, Dict
from datetime import date, timedelta
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config


class OpenFDAClient:
    """Client for openFDA NDC API."""
    
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = load_config()
        self.config = config
        self.base_url = config["apis"]["openfda"]["base_url"]
        self.timeout = config["apis"]["openfda"]["timeout_s"]
        self.api_key = os.getenv(config["apis"]["openfda"]["api_key_env"])
        self.cache_dir = Path(config["paths"]["api_cache_dir"]) / "openfda"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _make_request(self, params: Dict) -> Dict:
        """Make API request with rate limiting."""
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        
        response = requests.get(
            self.base_url,
            params=params,
            headers=headers,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def list_brand_ndcs_at_t0(
        self, 
        appl_no: str, 
        scd: Dict, 
        t0: date
    ) -> List[str]:
        """
        Return package_ndc list active at T0 for NDA appl_no and given SCD.
        
        Args:
            appl_no: Application number (e.g., "NDA021254")
            scd: Dict with ingredient, strength, dosage_form, route
            t0: Event date
        
        Returns:
            List of 11-digit package_ndc strings
        """
        # Format appl_no for openFDA (ensure NDA prefix)
        if not appl_no.startswith("NDA"):
            appl_no = f"NDA{appl_no.zfill(6)}"
        
        # Build search query
        # openFDA uses application_number field
        search_terms = [f'application_number:"{appl_no}"']
        
        # Add ingredient if available
        if scd.get("ingredient"):
            # Note: openFDA may use active_ingredient or product_type
            # We'll filter post-query for SCD matching
        
        params = {
            "search": "+".join(search_terms),
            "limit": 1000,  # Adjust if needed
        }
        
        try:
            data = self._make_request(params)
            results = data.get("results", [])
            
            # Filter by marketing dates and SCD
            brand_ndcs = []
            t0_str = t0.isoformat()
            
            for result in results:
                # Check marketing dates
                marketing_start = result.get("marketing_start_date")
                marketing_end = result.get("marketing_end_date")
                
                if marketing_start and marketing_start > t0_str:
                    continue
                if marketing_end and marketing_end < t0_str:
                    continue
                
                # Get package_ndc (11-digit)
                package_ndc = result.get("package_ndc")
                if not package_ndc:
                    continue
                
                # Normalize to 11 digits (remove dashes, pad if needed)
                package_ndc = package_ndc.replace("-", "").replace(" ", "")
                if len(package_ndc) == 10:
                    package_ndc = "0" + package_ndc
                elif len(package_ndc) != 11:
                    continue
                
                # TODO: Add SCD matching logic here if needed
                # For now, include all NDCs for the application
                brand_ndcs.append(package_ndc)
            
            return list(set(brand_ndcs))  # Deduplicate
        
        except requests.exceptions.RequestException as e:
            print(f"Error querying openFDA for {appl_no}: {e}")
            return []
    
    def list_generic_ndcs_by_t6(
        self, 
        scd: Dict, 
        t0: date
    ) -> Tuple[List[str], Set[str]]:
        """
        Return (package_ndc list, distinct labelers) for generics with 
        marketing_start ≤ T0+6m and matching SCD.
        
        Args:
            scd: Dict with ingredient, strength, dosage_form, route
            t0: Event date
        
        Returns:
            Tuple of (list of package_ndcs, set of labeler names)
        """
        t6 = t0 + timedelta(days=180)  # ~6 months
        t6_str = t6.isoformat()
        
        # Search for ANDA products
        # Note: openFDA uses marketing_category or product_type
        search_terms = ['marketing_category:"ANDA"']
        
        # Add ingredient search
        ingredient = scd.get("ingredient", "").upper()
        if ingredient:
            # openFDA may use active_ingredient field
            search_terms.append(f'active_ingredient:"{ingredient}"')
        
        params = {
            "search": "+".join(search_terms),
            "limit": 1000,
        }
        
        try:
            data = self._make_request(params)
            results = data.get("results", [])
            
            generic_ndcs = []
            labelers = set()
            
            for result in results:
                marketing_start = result.get("marketing_start_date")
                
                if not marketing_start or marketing_start > t6_str:
                    continue
                
                package_ndc = result.get("package_ndc")
                if not package_ndc:
                    continue
                
                package_ndc = package_ndc.replace("-", "").replace(" ", "")
                if len(package_ndc) == 10:
                    package_ndc = "0" + package_ndc
                elif len(package_ndc) != 11:
                    continue
                
                # Get labeler
                labeler = result.get("labeler_name") or result.get("manufacturer_name")
                if labeler:
                    labelers.add(labeler)
                
                generic_ndcs.append(package_ndc)
            
            return list(set(generic_ndcs)), labelers
        
        except requests.exceptions.RequestException as e:
            print(f"Error querying openFDA for generics: {e}")
            return [], set()


if __name__ == "__main__":
    # Sample usage
    client = OpenFDAClient()
    scd = {
        "ingredient": "ATORVASTATIN",
        "strength": "20 MG",
        "dosage_form": "TABLET",
        "route": "ORAL"
    }
    t0 = date(2019, 6, 15)
    ndcs, labelers = client.list_generic_ndcs_by_t6(scd, t0)
    print(f"Found {len(ndcs)} generic NDCs, {len(labelers)} labelers")

