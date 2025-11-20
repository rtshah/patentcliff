"""openFDA NDC API client for listing brand and generic NDCs."""
import requests
import os
import time
import re
from typing import List, Set, Tuple, Optional, Dict
from datetime import date, timedelta
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config

# Lucene special characters that need escaping
_LUCENE_ESC = re.compile(r'([+\-!(){}\[\]^"~*?:\\/])')


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
    
    def _escape_q(self, s: str) -> str:
        """
        Escape Lucene special characters in query string.
        
        Args:
            s: String to escape
        
        Returns:
            Escaped string
        """
        if not s:
            return ""
        s = _LUCENE_ESC.sub(r'\\\1', s)
        return " ".join(s.split())
    
    def _split_ingredients(self, raw: str) -> List[str]:
        """
        Split ingredient string on separators (;, +, /, comma).
        
        Args:
            raw: Raw ingredient string (e.g., "A; B" or "A + B")
        
        Returns:
            List of cleaned ingredient parts
        """
        if not raw:
            return []
        # Split on ; , + / and collapse whitespace
        parts = [p.strip() for p in re.split(r'[;,+/]', raw) if p.strip()]
        return parts
    
    def _norm_text(self, v) -> str:
        """
        Normalize text field (handles list, string, None).
        
        Args:
            v: Value (can be list, string, or None)
        
        Returns:
            Normalized uppercase string
        """
        if v is None:
            return ""
        if isinstance(v, list):
            v = ";".join(map(str, v))
        return str(v).upper().strip()
    
    def _make_request(self, params: Dict) -> Dict:
        """
        Make API request with rate limiting. Handles 404s gracefully.
        
        openFDA returns 404 when there are zero matches. Don't blow up—just return empty results.
        """
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        
        response = requests.get(
            self.base_url,
            params=params,
            headers=headers,
            timeout=self.timeout
        )
        
        # Treat 404 as no results (valid zero-hit case)
        if response.status_code == 404:
            return {"results": []}
        
        # Will raise on 4xx/5xx except 404 above
        response.raise_for_status()
        return response.json()
    
    def _to_upper_set(self, v) -> Set[str]:
        """
        Normalize a string/list/comma-string to a set of UPPER tokens.
        
        Args:
            v: String, list, or comma-separated string
        
        Returns:
            Set of uppercase tokens
        """
        if v is None:
            return set()
        if isinstance(v, list):
            items = v
        else:
            items = [p.strip() for p in str(v).split(",")]
        return {s.upper() for s in items if s}
    
    def _synonym_map(self) -> Dict[str, Set[str]]:
        """
        Very small synonym map to reduce false mismatches.
        
        Returns:
            Dict mapping canonical form to set of synonyms
        """
        return {
            # dosage forms
            "AEROSOL": {"AEROSOL", "SPRAY", "SPRAY, METERED"},
            "SPRAY": {"SPRAY", "SPRAY, METERED", "AEROSOL"},
            "INJECTABLE": {"INJECTABLE", "INJECTION", "SOLUTION", "SOLUTION FOR INJECTION"},
            "INJECTION": {"INJECTION", "INJECTABLE", "SOLUTION FOR INJECTION"},
            # routes
            "ORAL": {"ORAL"},
            "SUBLINGUAL": {"SUBLINGUAL"},
        }
    
    def _match_with_synonyms(self, item_vals: Set[str], scd_val: Optional[str]) -> bool:
        """
        Check if item values intersect with SCD value, with synonyms.
        
        Args:
            item_vals: Set of uppercase tokens from item
            scd_val: SCD value (string, may be comma-separated)
        
        Returns:
            True if there's a match (direct or via synonym)
        """
        if not scd_val:
            return True
        
        syn = self._synonym_map()
        target = scd_val.upper()
        target_set = syn.get(target, {target})
        
        # Also explode comma-joined SCD values into tokens
        scd_tokens = self._to_upper_set(scd_val)
        all_targets = scd_tokens | target_set
        
        # Expand targets with synonyms
        expanded_targets = set()
        for token in all_targets:
            expanded_targets.add(token)
            if token in syn:
                expanded_targets.update(syn[token])
        
        # Check if any item value matches any target (direct or synonym)
        return bool(item_vals & expanded_targets)
    
    def _build_search(
        self, 
        appl_no: Optional[str] = None, 
        scd: Optional[Dict] = None, 
        marketing_category: Optional[str] = None
    ) -> str:
        """
        Build minimal openFDA search string.
        
        Keep application_number (for brands) and marketing_category:"ANDA" (for generics).
        Use only active_ingredients.name for ingredient filter (split multi-ingredient combos).
        Filter dosage_form and route client-side (openFDA sometimes returns them as lists).
        
        Args:
            appl_no: Application number (e.g., "NDA021254" or "ANDA123456")
            scd: Dict with ingredient, strength, dosage_form, route
            marketing_category: Optional category filter (e.g., "ANDA" as hint)
        
        Returns:
            Search string for openFDA API
        """
        terms = []
        
        if appl_no:
            # Ensure NDA/ANDA prefix; zero-pad if needed
            up = appl_no.upper()
            if not (up.startswith("NDA") or up.startswith("ANDA")):
                up = f"NDA{str(appl_no).zfill(6)}"
            terms.append(f'application_number:"{self._escape_q(up)}"')
        
        if marketing_category:
            terms.append(f'marketing_category:"{self._escape_q(marketing_category)}"')
        
        if scd:
            ing = scd.get("ingredient") or ""
            parts = self._split_ingredients(ing)
            if parts:
                # Require all actives via AND (handles multi-ingredient combos)
                must_all = " AND ".join(
                    [f'active_ingredients.name:"{self._escape_q(p)}"' for p in parts]
                )
                terms.append(f"({must_all})")
        
        return " AND ".join(terms) if terms else ""
    
    def _normalize_ndc(self, ndc: str) -> Optional[str]:
        """
        Normalize NDC to 11-digit format.
        
        Args:
            ndc: NDC string (may contain dashes, spaces, or be 10/11 digits)
        
        Returns:
            11-digit NDC string or None if invalid
        """
        if not ndc:
            return None
        
        ndc = ndc.replace("-", "").replace(" ", "").strip()
        if len(ndc) == 10:
            ndc = "0" + ndc
        elif len(ndc) != 11:
            return None
        
        return ndc
    
    def _parse_date_safe(self, date_str: Optional[str]) -> Optional[int]:
        """
        Parse date string to YYYYMMDD integer for comparison.
        Handles both ISO format (YYYY-MM-DD) and YYYYMMDD format.
        
        Args:
            date_str: Date string in various formats
         
        Returns:
            Integer YYYYMMDD or None if invalid
        """
        if not date_str:
            return None
        
        # Remove any time component
        date_str = date_str.split("T")[0].split(" ")[0]
        
        # Try ISO format first (YYYY-MM-DD)
        if len(date_str) == 10 and date_str[4] == "-":
            try:
                return int(date_str.replace("-", ""))
            except ValueError:
                return None
        
        # Try YYYYMMDD format
        if len(date_str) == 8 and date_str.isdigit():
            try:
                return int(date_str)
            except ValueError:
                return None
        
        return None
    
    def list_brand_ndcs_at_t0(
        self, 
        appl_no: str, 
        scd: Dict, 
        t0: date
    ) -> List[str]:
        """
        Return package_ndc list active at T0 for NDA appl_no and given SCD.
        
        Uses minimal server-side search (ingredient only), then filters client-side.
        
        Args:
            appl_no: Application number (e.g., "NDA021254")
            scd: Dict with ingredient, strength, dosage_form, route
            t0: Event date
        
        Returns:
            List of 11-digit package_ndc strings
        """
        # Minimal server search: ingredient only
        search = self._build_search(appl_no=None, scd=scd, marketing_category=None)
        params = {"search": search, "limit": 1000}
        
        try:
            data = self._make_request(params)
            # Fallback: if empty, try ingredient-less (rarely needed)
            if not data.get("results"):
                data = self._make_request({"limit": 1000})
            
            ndcs = []
            t0_int = int(t0.strftime("%Y%m%d"))
            
            # Format target NDA for matching
            nda_pref = appl_no
            if not appl_no.startswith(("NDA", "ANDA")):
                nda_pref = f"NDA{appl_no.zfill(6)}"
            else:
                nda_pref = appl_no.upper()
            
            for item in data.get("results", []):
                # 1) Must be active at T0
                ms_int = self._parse_date_safe(item.get("marketing_start_date"))
                me_int = self._parse_date_safe(item.get("marketing_end_date"))
                
                if ms_int and ms_int > t0_int:
                    continue
                if me_int and me_int < t0_int:
                    continue
                
                # 2) Prefer NDA application_number (if present). Exclude ANDA.
                app_num = (item.get("application_number") or "").upper()
                if app_num.startswith("ANDA"):
                    continue
                
                # If NDA present and we have a target NDA, prefer match
                if nda_pref.startswith("NDA") and app_num.startswith("NDA"):
                    if nda_pref != app_num:
                        # Different NDA — skip
                        continue
                
                # 3) SCD check (use synonym matching for form/route)
                item_forms = self._to_upper_set(item.get("dosage_form"))
                item_routes = self._to_upper_set(item.get("route"))
                
                if not self._match_with_synonyms(item_forms, scd.get("dosage_form")):
                    continue
                if not self._match_with_synonyms(item_routes, scd.get("route")):
                    continue
                
                # 4) Gather package ndcs
                packaging = item.get("packaging") or []
                if packaging:
                    for pkg in packaging:
                        ndc = self._normalize_ndc(pkg.get("package_ndc") or "")
                        if ndc:
                            ndcs.append(ndc)
                else:
                    ndc = self._normalize_ndc(item.get("package_ndc") or "")
                    if ndc:
                        ndcs.append(ndc)
            
            return sorted(set(ndcs))
        
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
        
        Uses minimal server-side search (ingredient + ANDA hint), then filters client-side.
        
        Args:
            scd: Dict with ingredient, strength, dosage_form, route
            t0: Event date
        
        Returns:
            Tuple of (list of package_ndcs, set of labeler names)
        """
        t6 = t0 + timedelta(days=180)  # ~6 months
        t6_int = int(t6.strftime("%Y%m%d"))
        
        # Minimal server search: ingredient + ANDA hint
        search = self._build_search(appl_no=None, scd=scd, marketing_category="ANDA")
        params = {"search": search, "limit": 1000}
        
        try:
            data = self._make_request(params)
            # Fallback without marketing_category if needed
            if not data.get("results"):
                data = self._make_request({"search": self._build_search(appl_no=None, scd=scd), "limit": 1000})
            
            generic_ndcs = []
            labelers = set()
            
            for item in data.get("results", []):
                # 1) Ensure generic: application_number must start with ANDA
                app_num = (item.get("application_number") or "").upper()
                if not app_num.startswith("ANDA"):
                    continue
                
                # 2) Launched by T+6: marketing_start_date <= T6
                ms_int = self._parse_date_safe(item.get("marketing_start_date"))
                if not ms_int or ms_int > t6_int:
                    continue
                
                # 3) SCD check (use synonym matching for form/route)
                item_forms = self._to_upper_set(item.get("dosage_form"))
                item_routes = self._to_upper_set(item.get("route"))
                
                if not self._match_with_synonyms(item_forms, scd.get("dosage_form")):
                    continue
                if not self._match_with_synonyms(item_routes, scd.get("route")):
                    continue
                
                # 4) Get labeler
                labeler = (item.get("labeler_name") or item.get("manufacturer_name") or "").strip()
                if labeler:
                    labelers.add(labeler)
                
                # 5) Extract NDCs from packaging array or direct package_ndc field
                packaging = item.get("packaging") or []
                if packaging:
                    for pkg in packaging:
                        ndc = self._normalize_ndc(pkg.get("package_ndc") or "")
                        if ndc:
                            generic_ndcs.append(ndc)
                else:
                    ndc = self._normalize_ndc(item.get("package_ndc") or "")
                    if ndc:
                        generic_ndcs.append(ndc)
            
            return sorted(set(generic_ndcs)), labelers
        
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

