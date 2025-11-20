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
```bash
python -m src.label.build_label
```
**Note:** This step makes API calls to openFDA and NADAC. For initial testing with 50-100 events, this may take some time.

### Step 4: Build Features
```bash
python -m src.model.dataset
```

### Step 5: Train Models
```bash
python -m src.model.train
```

### Step 6: Generate Plots
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

### 4. Missing Data
- Some events may not have brand NDCs at T0 (flagged with `missing_brand_price_t0`)
- Some events may not have generics by T+6 (flagged with `missing_generic_price_t6`)
- These are filtered out in the QC step in `dataset.py`

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

