"""RxNorm API client for NDC to RxCUI and ATC mapping."""
import requests
from typing import Optional, Dict, List
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config


class RxNormClient:
    """Client for RxNorm/RxClass APIs."""
    
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = load_config()
        self.config = config
        self.rxcui_from_ndc_url = config["apis"]["rxnav"]["rxcui_from_ndc"]
        self.rxclass_from_rxcui_url = config["apis"]["rxnav"]["rxclass_from_rxcui"]
        self.cache_dir = Path(config["paths"]["api_cache_dir"]) / "rxnorm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def ndc_to_scd(self, ndc: str) -> Optional[str]:
        """
        Map NDC to RxNorm SCD (Semantic Clinical Drug) RxCUI.
        Returns RxCUI string or None.
        """
        # Normalize NDC (remove dashes)
        ndc_clean = ndc.replace("-", "").replace(" ", "")
        
        try:
            url = f"{self.rxcui_from_ndc_url}{ndc_clean}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Parse response structure
            # RxNorm API returns: {"ndcProperties": [{"rxcui": "...", ...}]}
            if "ndcProperties" in data and data["ndcProperties"]:
                return data["ndcProperties"][0].get("rxcui")
            
            return None
        except Exception as e:
            print(f"Error mapping NDC {ndc} to RxCUI: {e}")
            return None
    
    def scd_to_atc(self, rxcui: str) -> List[str]:
        """
        Map RxCUI to ATC codes via RxClass.
        Returns list of ATC codes.
        """
        try:
            url = self.rxclass_from_rxcui_url.format(rxcui=rxcui)
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Parse RxClass response
            atc_codes = []
            if "rxclassDrugInfoList" in data:
                for item in data["rxclassDrugInfoList"].get("rxclassDrugInfo", []):
                    if item.get("relaSource") == "ATC":
                        atc_codes.append(item.get("rxclassMinConceptItem", {}).get("classId"))
            
            return atc_codes
        except Exception as e:
            print(f"Error mapping RxCUI {rxcui} to ATC: {e}")
            return []


if __name__ == "__main__":
    client = RxNormClient()
    print("RxNorm client initialized")

