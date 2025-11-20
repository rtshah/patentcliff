"""SCD (RxNorm Semantic Clinical Drug) normalization utilities."""
import re
from typing import Dict, Optional, Tuple


def parse_df_route(df_route: str) -> Tuple[str, str]:
    """
    Split DF;Route field into dosage_form and route.
    Returns (dosage_form_text, route_text).
    """
    if not df_route or not isinstance(df_route, str):
        return "", ""
    
    parts = df_route.split(';', 1)
    dosage_form = parts[0].strip() if parts else ""
    route = parts[1].strip() if len(parts) > 1 else ""
    
    # Clean up whitespace
    dosage_form = re.sub(r'\s+', ' ', dosage_form).strip()
    route = re.sub(r'\s+', ' ', route).strip()
    
    return dosage_form, route


def build_scd_key(ingredient: str, strength: str, dosage_form: str, route: str) -> str:
    """
    Build a canonical SCD key for matching.
    Format: INGREDIENT|STRENGTH|DOSAGE_FORM|ROUTE
    """
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.utils import normalize_strength_for_matching, canonicalize_dosage_form, canonicalize_route
    
    ing = ingredient.upper().strip() if ingredient else ""
    str_norm = normalize_strength_for_matching(strength) if strength else ""
    df_canon = canonicalize_dosage_form(dosage_form) if dosage_form else ""
    rt_canon = canonicalize_route(route) if route else ""
    
    return "|".join([ing, str_norm, df_canon, rt_canon])


def extract_scd_from_product_row(row: Dict) -> Dict:
    """
    Extract SCD components from a product row dictionary.
    Returns dict with: ingredient, strength, dosage_form, route, scd_key
    """
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.utils import normalize_strength_for_matching, canonicalize_dosage_form, canonicalize_route
    
    ingredient = str(row.get("ingredient", "")).strip()
    strength = str(row.get("strength", "")).strip()
    df_route = str(row.get("df_route", "")).strip()
    
    dosage_form, route = parse_df_route(df_route)
    
    scd_key = build_scd_key(ingredient, strength, dosage_form, route)
    
    return {
        "ingredient": ingredient,
        "strength": strength,
        "strength_normalized": normalize_strength_for_matching(strength),
        "dosage_form": dosage_form,
        "dosage_form_canonical": canonicalize_dosage_form(dosage_form),
        "route": route,
        "route_canonical": canonicalize_route(route),
        "scd_key": scd_key,
    }

