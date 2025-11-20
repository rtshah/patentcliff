# Rx Price Erosion Prediction Pipeline

An end-to-end pipeline that predicts the percentage price drop six months after a brand drug loses patent/exclusivity protection.

## Overview

This project builds a machine learning model to predict pharmaceutical price erosion using:
- **Orange Book** data for patent/exclusivity expiry dates (T₀)
- **openFDA NDC API** for brand and generic product listings
- **NADAC** (National Average Drug Acquisition Cost) for pricing data
- **Medicare Part D** for market size/utilization features
- **RxNorm/RxClass** for therapeutic class mapping

## Project Structure

```
/proj
  /data/             # Raw Orange Book snapshots, API cache
  /config/            # config.yaml, cache/
  /artifacts/        # Events, labels, features, models
  /reports/          # Metrics, SHAP plots, figures
  /src/
    /connectors/     # API clients (openFDA, NADAC, Part D, RxNorm)
    /features/       # Feature engineering (complexity, volatility, SCD)
    /label/          # Event building, label construction
    /model/          # Training, ensemble, SHAP plots
    /service/        # FastAPI app
  /tests/            # Test suite
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure:**
   - Edit `config.yaml` to set paths and API endpoints
   - Orange Book data should be in `./data/EOBZIP_2025_10/` (or update path in config)

3. **Optional API keys:**
   ```bash
   export OPENFDA_API_KEY=your_key  # For higher rate limits
   export MEDICAID_APP_TOKEN=your_token
   export CMS_APP_TOKEN=your_token
   ```

## Usage

### Data Pipeline

```bash
# Discover NADAC UUID registry
make data

# Build events from Orange Book
make build_labels

# Build features
make features

# Train models
make train
```

### API Service

```bash
# Start FastAPI server
make serve
# or
uvicorn src.service.app:app --host 0.0.0.0 --port 8000
```

**Example request:**
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

### Docker

```bash
# Build image
docker build -t rx-price-erosion .

# Run container
docker run -p 8000:8000 rx-price-erosion
```

## Model Details

### Features
- `entrants_by_6m`: Number of generic labelers by T+6 months
- `complexity_score`: Route/form complexity (1-10)
- `patent_thickness`: Count of active patents at T₀
- `market_size_prior_year`: Part D gross spend
- `brand_price_volatility_12m`: Coefficient of variation of brand prices
- `calendar_y`, `calendar_q`: Temporal features

### Models
- **Baselines**: Class mean (by complexity), Linear regression
- **XGBoost**: Primary model with hyperparameter search
- **LightGBM**: Secondary tree-based model
- **MLP**: Tiny neural network (128→64→32→1)
- **Ensemble**: Weighted average optimized on validation RMSE

### Metrics
- RMSE, MAE, R² on test set (2018-2020)
- Stratified by complexity tertiles and market size tertiles

## Data Sources & Caveats

### Orange Book
- Monthly text/CSV exports from FDA
- Patent expiry dates may have format variations
- Exclusivity data optional (controlled by `use_exclusivity` flag)

### NADAC
- Medicaid DKAN API, one dataset per year
- UUID registry auto-discovered on first run
- Pricing units may vary (per-tab vs per-mL); mixed units flagged/dropped

### openFDA
- Public API, no authentication required (optional key for rate limits)
- Marketing dates approximate actual launch dates
- SCD matching relies on ingredient+strength+form+route

### Medicare Part D
- Excludes rebates (acceptable since target is acquisition cost)
- May require manual data pulls depending on API availability

### Generic Monthly NADAC
- Recent years may use 3-month moving averages (noted in reports)

## Testing

```bash
# Run test suite
make test

# Acceptance tests verify:
# - Join correctness (brand/generic NDC inclusion/exclusion)
# - Unit consistency (pricing unit normalization)
# - Label stability (missing price handling)
# - Temporal split integrity (no leakage)
# - API validation (Pydantic schemas)
```

## Outputs

- `artifacts/events.parquet`: T₀ events per NDA SCD
- `artifacts/labels.parquet`: Price targets with QC flags
- `artifacts/features.parquet`: Full feature table
- `artifacts/train.parquet`, `val.parquet`, `test.parquet`: Temporal splits
- `reports/metrics.json`: Model performance metrics
- `reports/shap_global.png`: Feature importance
- `reports/residuals_by_complexity.png`: Residual analysis
- `reports/pred_vs_actual.png`: Prediction scatter plot
- `mlruns/`: MLflow experiment tracking

## License

[Specify license]

## Contact

[Your contact information]

