"""Assemble feature table from labels and compute features."""
import pandas as pd
from pathlib import Path
from datetime import date
import sys
import json

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config
from src.features.complexity import complexity_score
from src.features.volatility import brand_price_cv_12m
from src.connectors.nadac_client import NADACClient
from src.connectors.openfda_client import OpenFDAClient
from src.connectors.partd_client import PartDClient
from src.connectors.rxnorm_client import RxNormClient


def _qc_flags(df: pd.DataFrame, config=None) -> pd.DataFrame:
    """
    Compute QC flags for diagnostic reporting.
    
    Args:
        df: Labels DataFrame
        config: Optional config dict (for QC thresholds)
    
    Returns:
        DataFrame with boolean flags for each QC check.
    """
    if config is None:
        config = load_config()
    
    qc_config = config.get("qc", {})
    
    def _lenlike(x):
        """Helper to count items in various formats."""
        if isinstance(x, (list, tuple, set)):
            return len(x)
        if isinstance(x, str):
            # Try to parse JSON-encoded array
            try:
                y = json.loads(x)
                return len(y) if isinstance(y, (list, tuple, set)) else (0 if not y else 1)
            except Exception:
                return 0 if x.strip() == "" else 1
        return 0 if x is None else 1
    
    flags = pd.DataFrame(index=df.index)
    
    # Year check: T0 must be within NADAC coverage (2013-2025)
    if "t0" in df.columns:
        t0_years = pd.to_datetime(df["t0"], errors="coerce").dt.year
        flags["qc_year_ok"] = t0_years.between(2013, 2025, inclusive="both")
    else:
        flags["qc_year_ok"] = True
    
    # Brand NDCs check
    if "brand_ndcs_count" in df.columns:
        flags["qc_has_brand"] = df["brand_ndcs_count"].fillna(0) > 0
    elif "brand_ndcs" in df.columns:
        flags["qc_has_brand"] = df["brand_ndcs"].apply(_lenlike) > 0
    else:
        flags["qc_has_brand"] = False
    
    # Generic NDCs check
    if "generic_ndcs_count" in df.columns:
        flags["qc_has_generic"] = df["generic_ndcs_count"].fillna(0) > 0
    elif "generic_ndcs" in df.columns:
        flags["qc_has_generic"] = df["generic_ndcs"].apply(_lenlike) > 0
    else:
        flags["qc_has_generic"] = False
    
    # NADAC price data check
    flags["qc_has_brand_price"] = ~df.get("missing_brand_price_t0", pd.Series(False, index=df.index))
    flags["qc_has_generic_price"] = ~df.get("missing_generic_price_t6", pd.Series(False, index=df.index))
    
    # Mixed units check
    flags["qc_no_mixed_units"] = ~df.get("mixed_units", pd.Series(False, index=df.index))
    
    # Generic labelers check
    min_labelers = qc_config.get("min_generic_labelers_t6", 1)
    if "entrants_by_6m" in df.columns:
        flags["qc_sufficient_labelers"] = df["entrants_by_6m"].fillna(0) >= min_labelers
    else:
        flags["qc_sufficient_labelers"] = False
    
    # Future date check
    flags["qc_not_future"] = ~df.get("t0_in_future", pd.Series(False, index=df.index))
    
    # Price drop computed: both prices present after unit harmonization & date snapping
    # (This means we can compute price_drop_pct, even if it's 0 or negative)
    flags["qc_has_price_drop"] = (
        ~df.get("missing_brand_price_t0", pd.Series(True, index=df.index)) &
        ~df.get("missing_generic_price_t6", pd.Series(True, index=df.index))
    )
    
    return flags


def assemble_training_table(labels_df: pd.DataFrame, config=None) -> pd.DataFrame:
    """
    Join labels + features; apply QC filters; return modeling DataFrame.
    """
    if config is None:
        config = load_config()
    
    # Compute QC flags for diagnostic reporting
    qc_flags = _qc_flags(labels_df, config)
    
    # Print QC pass rates (1.0 = 100% passing)
    print("\nQC pass rates (1.0 = 100% passing each gate):")
    pass_rates = qc_flags.mean().sort_values()
    for flag_name, pass_rate in pass_rates.items():
        print(f"  {flag_name}: {pass_rate:.3f} ({pass_rate*100:.1f}%)")
    
    # Show examples of failures for the worst-performing gates
    worst_gates = pass_rates.head(3)
    for flag_name in worst_gates.index:
        if pass_rates[flag_name] < 1.0:  # Only show if not 100% passing
            failing = labels_df.loc[~qc_flags[flag_name]].head(3)
            if len(failing) > 0:
                print(f"\nExamples failing {flag_name} (showing first 3):")
                display_cols = [c for c in ["appl_no", "scd_key", "t0", "ingredient", 
                                            "dosage_form", "route", "brand_ndcs_count", 
                                            "generic_ndcs_count", "entrants_by_6m",
                                            "missing_brand_price_t0", "missing_generic_price_t6",
                                            "price_drop_pct"] if c in failing.columns]
                print(failing[display_cols].to_string())
    
    # Apply QC filters
    qc_config = config["qc"]
    
    # Filter rows
    strict_df = labels_df.copy()
    
    # Drop if missing required prices
    strict_df = strict_df[
        ~strict_df["missing_brand_price_t0"] & 
        ~strict_df["missing_generic_price_t6"]
    ]
    
    # Drop if mixed units (if configured)
    if qc_config.get("drop_if_mixed_pricing_units", True):
        strict_df = strict_df[~strict_df["mixed_units"]]
    
    # Drop if too few generics
    min_labelers = qc_config.get("min_generic_labelers_t6", 1)
    strict_df = strict_df[strict_df["entrants_by_6m"] >= min_labelers]
    
    # Drop future dates
    strict_df = strict_df[~strict_df["t0_in_future"]]
    
    # Drop if price_drop_pct is missing
    strict_df = strict_df[strict_df["price_drop_pct"].notna()]
    
    print(f"\nAfter QC filters: {len(strict_df)} rows (from {len(labels_df)})")
    
    # Initialize clients for feature engineering
    nadac_client = NADACClient(config)
    openfda_client = OpenFDAClient(config)
    partd_client = PartDClient(config)
    rxnorm_client = RxNormClient(config) if config["features"]["use_rxnorm"] else None
    
    # Compute features
    features_list = []
    
    for idx, row in strict_df.iterrows():
        feat = {}
        
        # Basic features from labels
        feat["entrants_by_6m"] = row["entrants_by_6m"]
        feat["complexity_score"] = complexity_score(
            row.get("route_canonical", ""),
            row.get("dosage_form_canonical", ""),
            config
        )
        
        # Calendar features
        t0 = row["t0"]
        if isinstance(t0, str):
            t0 = pd.to_datetime(t0).date()
        feat["calendar_y"] = t0.year
        feat["calendar_q"] = (t0.month - 1) // 3 + 1
        
        # Patent thickness (placeholder - would need to count active patents at T0)
        # For now, set to 1 if we have a patent expiry
        feat["patent_thickness"] = 1 if row.get("t0") else 0
        
        # Volatility (requires brand NDCs - would need to fetch from openFDA)
        # Placeholder for now
        feat["brand_price_volatility_12m"] = 0.0
        
        # Market size (Part D - placeholder)
        feat["market_size_prior_year"] = 0.0
        feat["claims_prior_year"] = 0.0
        feat["beneficiaries_prior_year"] = 0.0
        
        # ATC codes (if RxNorm enabled)
        if rxnorm_client:
            # Would need to map NDC to RxCUI to ATC
            feat["atc_codes"] = []
        
        features_list.append(feat)
    
    features_df = pd.DataFrame(features_list)
    
    # Merge with labels
    result = pd.concat([strict_df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
    
    return result


def main():
    """Assemble training table and save."""
    config = load_config()
    
    # Load labels
    labels_path = Path(config["paths"]["artifacts_dir"]) / "labels.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}. Run build_label.py first.")
    
    labels = pd.read_parquet(labels_path)
    print(f"Loaded {len(labels)} labels")
    
    # Assemble features
    training_table = assemble_training_table(labels, config)
    
    # Temporal splits
    train_start = pd.to_datetime(config["modeling"]["splits"]["train_start"]).date()
    train_end = pd.to_datetime(config["modeling"]["splits"]["train_end"]).date()
    val_start = pd.to_datetime(config["modeling"]["splits"]["val_start"]).date()
    val_end = pd.to_datetime(config["modeling"]["splits"]["val_end"]).date()
    test_start = pd.to_datetime(config["modeling"]["splits"]["test_start"]).date()
    test_end = pd.to_datetime(config["modeling"]["splits"]["test_end"]).date()
    
    training_table["t0_date"] = pd.to_datetime(training_table["t0"]).dt.date
    
    train = training_table[
        (training_table["t0_date"] >= train_start) & 
        (training_table["t0_date"] <= train_end)
    ]
    val = training_table[
        (training_table["t0_date"] >= val_start) & 
        (training_table["t0_date"] <= val_end)
    ]
    test = training_table[
        (training_table["t0_date"] >= test_start) & 
        (training_table["t0_date"] <= test_end)
    ]
    
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Save
    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    training_table.to_parquet(artifacts_dir / "features.parquet", index=False)
    train.to_parquet(artifacts_dir / "train.parquet", index=False)
    val.to_parquet(artifacts_dir / "val.parquet", index=False)
    test.to_parquet(artifacts_dir / "test.parquet", index=False)
    
    print(f"Saved features to {artifacts_dir}")


if __name__ == "__main__":
    main()

