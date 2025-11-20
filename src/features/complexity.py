"""Compute complexity score from route and dosage form."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config


def complexity_score(route: str, form: str, config=None) -> int:
    """
    Compute complexity score from route and dosage form.
    Returns integer score (higher = more complex).
    """
    if config is None:
        config = load_config()
    
    complexity_map = config["features"]["complexity_map"]
    route_map = complexity_map.get("route", {})
    form_map = complexity_map.get("dosage_form", {})
    
    # Get scores
    route_score = route_map.get(route.upper(), 0)
    form_score = form_map.get(form.upper(), 0)
    
    # Return max of route and form (or sum, depending on design)
    # Using max for now as per spec
    return max(route_score, form_score)


if __name__ == "__main__":
    # Test
    score = complexity_score("ORAL", "TABLET")
    print(f"Complexity score for ORAL/TABLET: {score}")

