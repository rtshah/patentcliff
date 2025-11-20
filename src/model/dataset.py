"""Assemble feature table from labels and compute features."""
import pandas as pd
from pathlib import Path
from datetime import date
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config
from src.features.complexity import complexity_score
from src.features.volatility import brand_price_cv_12m
from src.connectors.nadac_client import NADACClient
from src.connectors.openfda_client import OpenFDAClient
from src.connectors.partd_client import PartDClient
from src.connectors.rxnorm_client import RxNormClient


def assemble_training_table(labels_df: pd.DataFrame, config=None) -> pd.DataFrame:
    """
    Join labels + features; apply QC filters; return modeling DataFrame.
    """
    if config is None:
        config = load_config()
    
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
    
    print(f"After QC filters: {len(strict_df)} rows (from {len(labels_df)})")
    
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

