"""Compute brand price volatility (CV) over a time window."""
import pandas as pd
from datetime import date, timedelta
from typing import List
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config
from src.connectors.nadac_client import NADACClient


def brand_price_cv_12m(
    brand_ndcs: List[str], 
    t0: date, 
    nadac_client: NADACClient,
    window_months: int = 12
) -> float:
    """
    Compute coefficient of variation (CV) of brand NADAC prices 
    over [T0 - window_months, T0 - 1].
    
    Returns CV as float, or 0.0 if insufficient data.
    """
    if not brand_ndcs:
        return 0.0
    
    # Collect prices over the window
    all_prices = []
    
    for month_offset in range(1, window_months + 1):
        target_date = t0 - timedelta(days=30 * month_offset)
        year_month = target_date.strftime("%Y-%m")
        
        nadac_data = nadac_client.fetch_nadac_month(brand_ndcs, year_month)
        
        if not nadac_data.empty and "nadac_per_unit" in nadac_data.columns:
            prices = pd.to_numeric(nadac_data["nadac_per_unit"], errors="coerce").dropna()
            if not prices.empty:
                # Use median per month
                all_prices.append(float(prices.median()))
    
    if len(all_prices) < 2:
        return 0.0
    
    prices_series = pd.Series(all_prices)
    mean_price = prices_series.mean()
    
    if mean_price == 0:
        return 0.0
    
    cv = prices_series.std() / mean_price
    return float(cv)


if __name__ == "__main__":
    from src.connectors.nadac_client import NADACClient
    from src.utils import load_config
    
    config = load_config()
    client = NADACClient(config)
    t0 = date(2019, 6, 15)
    ndcs = ["12345678901"]  # Example
    cv = brand_price_cv_12m(ndcs, t0, client)
    print(f"CV: {cv}")

