#!/usr/bin/env python3
"""Standalone QC diagnostic script - no code changes to pipeline needed."""
from pathlib import Path
import pandas as pd
import json
import sys

# Find the most recent labels file under artifacts/
root = Path.cwd()
cands = sorted(list((root / "artifacts").rglob("labels.*")), key=lambda p: p.stat().st_mtime)
if not cands:
    print("No labels.* under artifacts/")
    sys.exit(0)

labels_path = cands[-1]
print(f"Reading: {labels_path}")

# Load
if labels_path.suffix == ".parquet":
    df = pd.read_parquet(labels_path)
else:
    df = pd.read_csv(labels_path)

print(f"Rows: {len(df)}\nColumns: {list(df.columns)}\n")

# Helpers
def _lenlike(x):
    if isinstance(x, (list, tuple, set)):
        return len(x)
    if isinstance(x, str):
        # JSON-encoded array? try to parse
        try:
            y = json.loads(x)
            return len(y) if isinstance(y, (list, tuple, set)) else (0 if not y else 1)
        except Exception:
            return 0 if x.strip() == "" else 1
    return 0 if x is None else 1

def _to_year(col):
    s = pd.to_datetime(df[col], errors="coerce")
    return s.dt.year

# Build QC flags (adjust names if yours differ)
has_brand = df.get("brand_ndcs_count") if "brand_ndcs_count" in df.columns else df.get("brand_ndcs")
has_generic = df.get("generic_ndcs_count") if "generic_ndcs_count" in df.columns else df.get("generic_ndcs")

qc = pd.DataFrame(index=df.index)

# Year check
qc["qc_year_ok"] = _to_year("t0").between(2013, 2025, inclusive="both") if "t0" in df.columns else pd.Series(True, index=df.index)

# Brand check
if has_brand is not None:
    if pd.api.types.is_numeric_dtype(has_brand):
        qc["qc_has_brand"] = has_brand.fillna(0) > 0
    else:
        qc["qc_has_brand"] = has_brand.apply(_lenlike) > 0
else:
    qc["qc_has_brand"] = False

# Generic check
if has_generic is not None:
    if pd.api.types.is_numeric_dtype(has_generic):
        qc["qc_has_generic"] = has_generic.fillna(0) > 0
    else:
        qc["qc_has_generic"] = has_generic.apply(_lenlike) > 0
else:
    qc["qc_has_generic"] = False

# Price checks
qc["qc_has_brand_price"] = ~df.get("missing_brand_price_t0", pd.Series(False, index=df.index))
qc["qc_has_generic_price"] = ~df.get("missing_generic_price_t6", pd.Series(False, index=df.index))

# Other checks
mixed_units_col = df.get("mixed_units", pd.Series(False, index=df.index))
qc["qc_no_mixed_units"] = ~mixed_units_col.fillna(False)
qc["qc_sufficient_labelers"] = df.get("entrants_by_6m", pd.Series(0, index=df.index)).fillna(0) >= 1
t0_in_future_col = df.get("t0_in_future", pd.Series(False, index=df.index))
qc["qc_not_future"] = ~t0_in_future_col.fillna(False)
# Price drop computed: both prices present after unit harmonization & date snapping
qc["qc_has_price_drop"] = (
    ~df.get("missing_brand_price_t0", pd.Series(True, index=df.index)) &
    ~df.get("missing_generic_price_t6", pd.Series(True, index=df.index))
)

# Print pass rates
print("QC pass rates (1.0 = 100% passing each gate):")
print(qc.mean().sort_values())
print()

# Coverage report by route/form
print("=" * 70)
print("COVERAGE REPORT BY ROUTE/FORM")
print("=" * 70)
print()

if "route_canonical" in df.columns and "dosage_form_canonical" in df.columns:
    coverage_report = df.groupby(["route_canonical", "dosage_form_canonical"]).agg({
        "brand_ndcs_count": lambda x: (x > 0).sum(),
        "generic_ndcs_count": lambda x: (x > 0).sum(),
        "price_t0": lambda x: x.notna().sum(),
        "price_t6": lambda x: x.notna().sum(),
        "no_nadac_coverage_t0": lambda x: x.sum() if "no_nadac_coverage_t0" in df.columns else 0,
        "no_nadac_coverage_t6": lambda x: x.sum() if "no_nadac_coverage_t6" in df.columns else 0,
    }).rename(columns={
        "brand_ndcs_count": "has_brand_ndcs",
        "generic_ndcs_count": "has_generic_ndcs",
        "price_t0": "has_price_t0",
        "price_t6": "has_price_t6",
        "no_nadac_coverage_t0": "no_coverage_t0",
        "no_nadac_coverage_t6": "no_coverage_t6",
    })
    
    # Add total count and has_both_prices
    coverage_report["total"] = df.groupby(["route_canonical", "dosage_form_canonical"]).size()
    coverage_report["has_both_prices"] = (
        df.groupby(["route_canonical", "dosage_form_canonical"])
        .apply(lambda g: ((g["price_t0"].notna()) & (g["price_t6"].notna())).sum())
    )
    
    # Sort by total descending
    coverage_report = coverage_report.sort_values("total", ascending=False)
    
    print(coverage_report.to_string())
    print()
    
    # Retail-only vs all comparison
    if "retail_filter_excluded" in df.columns:
        retail_only = df[df.get("retail_filter_excluded", pd.Series(False, index=df.index)) == False]
        print("=" * 70)
        print("RETAIL-ONLY vs ALL COMPARISON")
        print("=" * 70)
        print()
        print(f"All events: {len(df)}")
        print(f"Retail-only (after filter): {len(retail_only)}")
        print()
        
        print("QC pass rates - ALL:")
        print(qc.mean().sort_values())
        print()
        
        if len(retail_only) > 0:
            qc_retail = qc.loc[retail_only.index]
            print("QC pass rates - RETAIL-ONLY:")
            print(qc_retail.mean().sort_values())
            print()
    else:
        print("(Retail filter not applied - retail_filter_excluded column not found)")
        print()

# Show first few failures per gate
for col in qc.columns:
    bad = df.loc[~qc[col]].head(5)
    if len(bad) > 0:
        keep = [x for x in ["appl_no", "scd_key", "t0", "ingredient", "dosage_form", "route", 
                           "brand_ndcs_count", "generic_ndcs_count", "entrants_by_6m",
                           "missing_brand_price_t0", "missing_generic_price_t6", "price_drop_pct"] 
                if x in df.columns]
        print(f"\nExamples failing {col} (showing first 5):")
        print(bad[keep].to_string())
        print()

