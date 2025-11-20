"""Pydantic schemas for FastAPI."""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
from datetime import date


class PredictRequest(BaseModel):
    """Request schema for /predict endpoint."""
    ingredient: str = Field(..., description="Active ingredient name")
    strength: str = Field(..., description="Strength (e.g., '20 mg')")
    dosage_form: str = Field(..., description="Dosage form (e.g., 'tablet')")
    route: str = Field(..., description="Route of administration (e.g., 'oral')")
    expiry_date: str = Field(..., description="Patent/exclusivity expiry date (YYYY-MM-DD)")
    patent_thickness: int = Field(..., ge=0, description="Number of patents active at T0")
    entrants_by_6m: int = Field(..., ge=0, description="Number of generic labelers by T+6 months")
    complexity_score: int = Field(..., ge=0, le=10, description="Complexity score (0-10)")
    market_size_prior_year: float = Field(..., ge=0, description="Prior year market size (Part D spend)")
    brand_price_volatility_12m: float = Field(..., ge=0, le=1, description="Brand price CV over 12 months")
    
    @validator("expiry_date")
    def validate_date(cls, v):
        try:
            date.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError("expiry_date must be in YYYY-MM-DD format")


class Driver(BaseModel):
    """Feature driver with impact."""
    feature: str
    direction: str = Field(..., description="'up' or 'down'")
    impact: float = Field(..., ge=0, le=1, description="Relative impact (0-1)")


class PredictResponse(BaseModel):
    """Response schema for /predict endpoint."""
    predicted_price_drop_pct: float = Field(..., description="Predicted % price drop")
    top_drivers: List[Driver] = Field(..., description="Top feature drivers")
    model_version: str = Field(default="v1.0.0", description="Model version")

