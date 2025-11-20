# Quick Start Guide

## Initial Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Orange Book data:**
   - Ensure `./data/EOBZIP_2025_10/` contains:
     - `products.txt`
     - `patent.txt`
     - `exclusivity.txt` (optional)

3. **Test Orange Book parsing:**
   ```bash
   python -m src.label.build_events
   ```
   This should create `artifacts/events.parquet` and `artifacts/events_sample.csv`

## Running the Pipeline

### Step 1: Discover NADAC Registry
```bash
python -m src.connectors.nadac_client --discover-registry
```
This queries the Medicaid DKAN catalog and saves UUID mappings to `config/cache/nadac_uuid_registry.json`

### Step 2: Build Events
```bash
python -m src.label.build_events
```

### Step 3: Build Labels (requires API access)

**Full run (all events):**
```bash
python -m src.label.build_label
```

**Quick development run (100 events only):**
```bash
python -m src.label.build_label --limit 100
```

**Or use Makefile shortcut:**
```bash
make build_labels_quick
```

**Note:** This step makes API calls to openFDA and NADAC. For initial testing, use `--limit 100` to process only 100 events for faster iteration.

### Step 4: Diagnose QC Issues (if needed)

If you see "After QC filters: 0 rows", run the diagnostic script:
```bash
python scripts/diagnose_qc.py
```

This will show:
- QC pass rates for each gate (1.0 = 100% passing)
- Examples of rows failing each QC check
- Helps identify which filter is too strict

### Step 5: Build Features
```bash
python -m src.model.dataset
```

**Note:** The dataset script now includes QC diagnostics automatically. It will print pass rates and examples before filtering.

### Step 6: Train Models
```bash
python -m src.model.train
```

### Step 7: Generate Plots
```bash
python -m src.model.shap_plots
```

## API Service

Start the FastAPI server:
```bash
uvicorn src.service.app:app --host 0.0.0.0 --port 8000 --reload
```

Test the endpoint:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ingredient": "atorvastatin",
    "strength": "20 mg",
    "dosage_form": "tablet",
    "route": "oral",
    "expiry_date": "2019-06-15",
    "patent_thickness": 6,
    "entrants_by_6m": 5,
    "complexity_score": 1,
    "market_size_prior_year": 125.4,
    "brand_price_volatility_12m": 0.11
  }'
```

## Common Issues

### 1. Import Errors
If you see import errors, ensure you're running from the project root:
```bash
cd /Users/rahulshah/Desktop/generics_proj/proj
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 2. NADAC API Issues
- The NADAC UUID registry discovery may fail if the Medicaid DKAN API structure has changed
- Check `config/cache/nadac_uuid_registry.json` and manually add year→UUID mappings if needed
- Update `config.yaml` with manual overrides in `apis.medicaid_dkan.nadac_uuid_registry`

### 3. openFDA Rate Limits
- Add `OPENFDA_API_KEY` environment variable for higher rate limits
- The client includes basic rate limiting, but you may need to add delays for large batches

### 4. Missing Data / QC Filtering Issues

**If you see "After QC filters: 0 rows":**

1. **Run diagnostics:**
   ```bash
   python scripts/diagnose_qc.py
   ```

2. **Common issues and fixes:**
   
   **qc_year_ok = 0.0 (T0 dates outside NADAC coverage 2013-2025)**
   - Many events have T0 dates in the future (2026-2034)
   - These can't have NADAC prices since NADAC only covers 2013-2025
   - **Fix:** Filter events to T0 dates within NADAC window before building labels
   
   **qc_has_brand = 0.0 (No brand NDCs found)**
   - openFDA queries may be too strict or failing on combo ingredients
   - **Fix:** Check openFDA client logs for 400 errors on combo ingredients
   
   **qc_has_brand_price = 0.0 (No brand prices)**
   - Usually because T0 dates are future/out-of-range OR no brand NDCs found
   - **Fix:** Ensure T0 dates are within NADAC coverage window
   
   **qc_has_generic_price = 0.0 (No generic prices)**
   - Usually because T+6 months falls outside NADAC coverage OR no generics found
   - **Fix:** Ensure T0 dates allow T+6 to be within NADAC window

3. **Some events may not have brand NDCs at T0** (flagged with `missing_brand_price_t0`)
4. **Some events may not have generics by T+6** (flagged with `missing_generic_price_t6`)
5. **These are filtered out in the QC step in `dataset.py`**

## Next Steps

1. **Test with small sample:** Start with 10-20 events to verify the pipeline
2. **Review QC flags:** Check `artifacts/labels_sample.csv` for data quality issues
3. **Adjust features:** Modify `src/model/dataset.py` to add/remove features
4. **Tune models:** Adjust hyperparameters in `config.yaml` under `modeling.xgb.param_space`
5. **Add tests:** Expand `tests/` with mocked API responses for faster iteration

## File Structure Reference

- `src/label/orange_book_parser.py`: Parses Orange Book files, computes T0
- `src/connectors/openfda_client.py`: Queries openFDA for NDC listings
- `src/connectors/nadac_client.py`: Fetches NADAC pricing data
- `src/label/build_label.py`: Computes price_T0, price_T6, price_drop_pct
- `src/model/dataset.py`: Assembles feature table, applies QC filters
- `src/model/train.py`: Trains all models, logs to MLflow
- `src/service/app.py`: FastAPI service

