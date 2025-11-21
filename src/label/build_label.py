"""Build labels (price_T0, price_T6, price_drop_pct) from events."""
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from typing import Dict, Optional
import sys
import argparse

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config
from src.connectors.openfda_client import OpenFDAClient
from src.connectors.nadac_client import NADACClient


def build_label_row(event: Dict, openfda_client: OpenFDAClient, nadac_client: NADACClient, config: Optional[Dict] = None) -> Dict:
    """
    Build label row for a single event.
    
    Input: {appl_no, scd_key, ingredient, strength, dosage_form, route, t0, ...}
    Output: {
      price_t0, price_t6, price_drop_pct,
      entrants_by_6m, 
      qc: {missing_brand_price_t0, missing_generic_price_t6, mixed_units, ...}
    }
    """
    if config is None:
        config = load_config()
    
    # Retail filter check
    retail_config = config.get("retail_filter", {})
    if retail_config.get("enabled", False):
        route_canonical = event.get("route_canonical", event.get("route", "")).upper()
        dosage_form_canonical = event.get("dosage_form_canonical", event.get("dosage_form", "")).upper()
        
        allowed_routes = {r.upper() for r in retail_config.get("allowed_routes", [])}
        allowed_forms = {f.upper() for f in retail_config.get("allowed_dosage_forms", [])}
        
        if route_canonical not in allowed_routes or dosage_form_canonical not in allowed_forms:
            # Return early with retail_filter_excluded flag
            return {
                **event,
                "price_t0": None,
                "price_t6": None,
                "price_drop_pct": None,
                "entrants_by_6m": 0,
                "brand_ndcs_count": 0,
                "generic_ndcs_count": 0,
                "pricing_unit_t0": None,
                "pricing_unit_t6": None,
                "missing_brand_price_t0": True,
                "missing_generic_price_t6": True,
                "no_nadac_coverage_t0": True,
                "no_nadac_coverage_t6": True,
                "retail_filter_excluded": True,
            }
    
    t0 = event["t0"]
    if isinstance(t0, str):
        t0 = pd.to_datetime(t0).date()
    elif isinstance(t0, pd.Timestamp):
        t0 = t0.date()
    elif hasattr(t0, 'date'):  # Handle other datetime-like objects
        t0 = t0.date()
    
    scd = {
        "ingredient": event.get("ingredient", ""),
        "strength": event.get("strength", ""),
        "dosage_form": event.get("dosage_form", ""),
        "route": event.get("route", ""),
    }
    
    # Get brand NDCs at T0
    brand_ndcs = openfda_client.list_brand_ndcs_at_t0(
        event["appl_no"], 
        scd, 
        t0
    )
    
    # Get generic NDCs by T+6
    generic_ndcs, labelers = openfda_client.list_generic_ndcs_by_t6(scd, t0)
    entrants_by_6m = len(labelers)
    
    # Compute price_T0 with date snapping (and optional NDC-9 fallback)
    t0_year_month = t0.strftime("%Y-%m")
    snap_days = config.get("nadac", {}).get("snap_days", 45)
    use_ndc9_fallback = config.get("nadac", {}).get("use_ndc9_fallback", False)
    
    if use_ndc9_fallback:
        brand_nadac, brand_join_flags = nadac_client.fetch_nadac_with_fallback(
            brand_ndcs, t0_year_month, snap_days=snap_days
        )
        joined_on_ndc11_t0 = brand_join_flags.any() if len(brand_join_flags) > 0 else False
    else:
        brand_nadac = nadac_client.fetch_nadac_month(brand_ndcs, t0_year_month, snap_days=snap_days)
        joined_on_ndc11_t0 = True  # Assume NDC-11 when fallback disabled
    
    # Coverage gap detection: check if any NDCs mapped to NADAC
    brand_ndcs_in_nadac = set(brand_nadac["ndc"].unique()) if not brand_nadac.empty and "ndc" in brand_nadac.columns else set()
    no_nadac_coverage_t0 = len(brand_ndcs) > 0 and len(brand_ndcs_in_nadac) == 0
    
    price_t0 = None
    pricing_unit_t0 = None
    missing_brand_price_t0 = True
    
    if not brand_nadac.empty and "nadac_per_unit" in brand_nadac.columns:
        # Check for mixed units
        units = brand_nadac["pricing_unit"].dropna().unique()
        if len(units) > 1:
            # Mixed units - flag it
            mixed_units_t0 = True
        else:
            mixed_units_t0 = False
            pricing_unit_t0 = units[0] if len(units) > 0 else None
        
        # Compute median price
        prices = pd.to_numeric(brand_nadac["nadac_per_unit"], errors="coerce").dropna()
        if not prices.empty:
            price_t0 = float(prices.median())
            missing_brand_price_t0 = False
    else:
        mixed_units_t0 = False
    
    # Compute price_T6 with date snapping (and optional NDC-9 fallback)
    t6 = t0 + timedelta(days=180)  # ~6 months
    t6_year_month = t6.strftime("%Y-%m")
    
    if use_ndc9_fallback:
        generic_nadac, generic_join_flags = nadac_client.fetch_nadac_with_fallback(
            generic_ndcs, t6_year_month, snap_days=snap_days
        )
        joined_on_ndc11_t6 = generic_join_flags.any() if len(generic_join_flags) > 0 else False
    else:
        generic_nadac = nadac_client.fetch_nadac_month(generic_ndcs, t6_year_month, snap_days=snap_days)
        joined_on_ndc11_t6 = True  # Assume NDC-11 when fallback disabled
    
    # Coverage gap detection: check if any NDCs mapped to NADAC
    generic_ndcs_in_nadac = set(generic_nadac["ndc"].unique()) if not generic_nadac.empty and "ndc" in generic_nadac.columns else set()
    no_nadac_coverage_t6 = len(generic_ndcs) > 0 and len(generic_ndcs_in_nadac) == 0
    
    price_t6 = None
    pricing_unit_t6 = None
    missing_generic_price_t6 = True
    
    if not generic_nadac.empty and "nadac_per_unit" in generic_nadac.columns:
        units = generic_nadac["pricing_unit"].dropna().unique()
        if len(units) > 1:
            mixed_units_t6 = True
        else:
            mixed_units_t6 = False
            pricing_unit_t6 = units[0] if len(units) > 0 else None
        
        prices = pd.to_numeric(generic_nadac["nadac_per_unit"], errors="coerce").dropna()
        if not prices.empty:
            price_t6 = float(prices.median())
            missing_generic_price_t6 = False
    else:
        mixed_units_t6 = False
    
    # Compute price_drop_pct
    price_drop_pct = None
    if price_t0 is not None and price_t6 is not None and price_t0 > 0:
        price_drop_pct = ((price_t0 - price_t6) / price_t0) * 100
    
    # QC flags
    mixed_units = mixed_units_t0 or mixed_units_t6
    too_few_generic_ndcs = len(generic_ndcs) < 2
    t0_in_future = t0 > date.today()
    
    result = {
        **event,  # Include all original event fields
        "price_t0": price_t0,
        "price_t6": price_t6,
        "price_drop_pct": price_drop_pct,
        "entrants_by_6m": entrants_by_6m,
        "brand_ndcs_count": len(brand_ndcs),
        "generic_ndcs_count": len(generic_ndcs),
        "pricing_unit_t0": pricing_unit_t0,
        "pricing_unit_t6": pricing_unit_t6,
        "missing_brand_price_t0": missing_brand_price_t0,
        "missing_generic_price_t6": missing_generic_price_t6,
        "no_nadac_coverage_t0": no_nadac_coverage_t0,
        "no_nadac_coverage_t6": no_nadac_coverage_t6,
        "joined_on_ndc11_t0": joined_on_ndc11_t0,
        "joined_on_ndc11_t6": joined_on_ndc11_t6,
        "mixed_units": mixed_units,
        "too_few_generic_ndcs": too_few_generic_ndcs,
        "t0_in_future": t0_in_future,
        "retail_filter_excluded": False,
    }
    
    return result


def main(limit: Optional[int] = None):
    """
    Build labels.parquet from events.
    
    Args:
        limit: Optional limit on number of events to process (for faster development/testing)
    """
    config = load_config()
    
    # Load events
    events_path = Path(config["paths"]["artifacts_dir"]) / "events.parquet"
    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}. Run build_events.py first.")
    
    events = pd.read_parquet(events_path)
    print(f"Loaded {len(events)} events")
    
    # Filter to events with T0 dates that allow NADAC prices
    # NADAC coverage: 2013-2025, need T0 and T0+6m to be within window
    events["t0"] = pd.to_datetime(events["t0"])
    nadac_max_date = pd.Timestamp("2025-12-31")
    cutoff = nadac_max_date - pd.Timedelta(days=180)  # 6 months before max date
    
    events = events[
        (events["t0"] <= cutoff) & 
        (events["t0"].dt.year.between(2013, 2025, inclusive="both"))
    ]
    print(f"After filtering to NADAC window (2013-2025, T0+6m <= 2025-12-31): {len(events)} events")
    
    # Limit events if specified (for faster development/testing)
    if limit is not None and limit > 0:
        events = events.head(limit)
        print(f"Limiting to first {len(events)} events for faster iteration")
    
    # Initialize clients
    openfda_client = OpenFDAClient(config)
    nadac_client = NADACClient(config)
    
    # Discover NADAC registry if needed
    if not nadac_client.uuid_registry:
        print("Discovering NADAC UUID registry...")
        nadac_client.discover_uuid_registry()
    
    # Build labels
    labels = []
    total_events = len(events)
    for idx, event in events.iterrows():
        print(f"Processing event {idx+1}/{total_events} ({event.get('ingredient', 'unknown')})...")
        
        try:
            label_row = build_label_row(event.to_dict(), openfda_client, nadac_client, config)
            labels.append(label_row)
            print(f"  ✓ Completed event {idx+1}")
        except Exception as e:
            print(f"  ✗ Error processing event {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    labels_df = pd.DataFrame(labels)
    
    # Save
    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    output_path = artifacts_dir / "labels.parquet"
    labels_df.to_parquet(output_path, index=False)
    print(f"Saved {len(labels_df)} labels to {output_path}")
    
    # Save sample
    labels_df.head(100).to_csv(artifacts_dir / "labels_sample.csv", index=False)
    print(f"Saved sample to {artifacts_dir / 'labels_sample.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build labels from events")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of events to process (for faster development/testing). Example: --limit 100"
    )
    args = parser.parse_args()
    main(limit=args.limit)

