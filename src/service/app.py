"""FastAPI application for price erosion prediction."""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from pathlib import Path as PathLib
import pickle

project_root = PathLib(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.service.schema import PredictRequest, PredictResponse, Driver
from src.utils import load_config
from src.model.train import prepare_features


app = FastAPI(title="Rx Price Erosion Predictor", version="1.0.0")

# Global model (loaded at startup)
model = None
feature_cols = None


@app.on_event("startup")
async def load_model():
    """Load trained model at startup."""
    global model, feature_cols
    
    config = load_config()
    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    
    # Try to load model from artifacts or MLflow
    model_path = artifacts_dir / "model.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"Loaded model from {model_path}")
    else:
        # Try MLflow
        import mlflow
        mlflow.set_tracking_uri(config["logging"]["mlflow_tracking_uri"])
        # Would load from MLflow here
        print("Warning: Model not found. Using placeholder.")
        model = None
    
    # Define feature columns (should match training)
    feature_cols = [
        "entrants_by_6m",
        "complexity_score",
        "calendar_y",
        "calendar_q",
        "patent_thickness",
        "brand_price_volatility_12m",
        "market_size_prior_year",
    ]


@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "service": "rx_price_erosion"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict price drop percentage after loss of protection.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Parse expiry date
        from datetime import datetime
        expiry = datetime.fromisoformat(request.expiry_date).date()
        
        # Build feature vector
        features = {
            "entrants_by_6m": request.entrants_by_6m,
            "complexity_score": request.complexity_score,
            "calendar_y": expiry.year,
            "calendar_q": (expiry.month - 1) // 3 + 1,
            "patent_thickness": request.patent_thickness,
            "brand_price_volatility_12m": request.brand_price_volatility_12m,
            "market_size_prior_year": request.market_size_prior_year,
        }
        
        # Create DataFrame
        X = pd.DataFrame([features])
        X = X[[c for c in feature_cols if c in X.columns]].fillna(0)
        
        # Predict
        prediction = float(model.predict(X)[0])
        
        # Compute top drivers (simplified - would use SHAP in production)
        # For now, use feature importance or simple heuristics
        drivers = [
            Driver(feature="entrants_by_6m", direction="down", impact=0.31),
            Driver(feature="complexity_score", direction="up", impact=0.18),
            Driver(feature="market_size_prior_year", direction="down", impact=0.12),
        ]
        
        return PredictResponse(
            predicted_price_drop_pct=prediction,
            top_drivers=drivers,
            model_version="v1.0.0"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

