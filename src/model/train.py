"""Train models: baselines, XGBoost, LightGBM, MLP, ensemble."""
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
from typing import Dict, Tuple
import sys
from pathlib import Path as PathLib

project_root = PathLib(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config


class TinyMLP(nn.Module):
    """Tiny MLP: 128→64→32→1"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.net(x)


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target."""
    feature_cols = [
        "entrants_by_6m",
        "complexity_score",
        "calendar_y",
        "calendar_q",
        "patent_thickness",
        "brand_price_volatility_12m",
        "market_size_prior_year",
    ]
    
    # Select available features
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].fillna(0)
    y = df["price_drop_pct"]
    
    return X, y


def baseline_class_mean(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Dict:
    """Class mean baseline: predict mean within route/form group."""
    X_train, y_train = prepare_features(train)
    X_val, y_val = prepare_features(val)
    X_test, y_test = prepare_features(test)
    
    # Group by complexity_score (or route/form)
    train = train.copy()
    val = val.copy()
    test = test.copy()
    
    train["pred"] = train.groupby("complexity_score")["price_drop_pct"].transform("mean")
    val["pred"] = val.groupby("complexity_score")["price_drop_pct"].transform("mean")
    test["pred"] = test.groupby("complexity_score")["price_drop_pct"].transform("mean")
    
    # Fill missing with overall mean
    overall_mean = train["price_drop_pct"].mean()
    train["pred"] = train["pred"].fillna(overall_mean)
    val["pred"] = val["pred"].fillna(overall_mean)
    test["pred"] = test["pred"].fillna(overall_mean)
    
    metrics = {
        "train_rmse": np.sqrt(mean_squared_error(y_train, train["pred"])),
        "val_rmse": np.sqrt(mean_squared_error(y_val, val["pred"])),
        "test_rmse": np.sqrt(mean_squared_error(y_test, test["pred"])),
        "test_mae": mean_absolute_error(y_test, test["pred"]),
        "test_r2": r2_score(y_test, test["pred"]),
    }
    
    return {"model": "class_mean", "metrics": metrics, "predictions": test["pred"]}


def baseline_linear(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Dict:
    """Linear regression baseline."""
    X_train, y_train = prepare_features(train)
    X_val, y_val = prepare_features(val)
    X_test, y_test = prepare_features(test)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    metrics = {
        "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
        "val_rmse": np.sqrt(mean_squared_error(y_val, val_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
        "test_mae": mean_absolute_error(y_test, test_pred),
        "test_r2": r2_score(y_test, test_pred),
    }
    
    return {"model": "linear", "metrics": metrics, "predictions": test_pred, "model_obj": model}


def train_xgboost(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, config: Dict) -> Dict:
    """Train XGBoost with hyperparameter search."""
    X_train, y_train = prepare_features(train)
    X_val, y_val = prepare_features(val)
    X_test, y_test = prepare_features(test)
    
    # Simple hyperparameter search (randomized)
    n_trials = config["modeling"]["xgb"]["n_trials"]
    param_space = config["modeling"]["xgb"]["param_space"]
    
    best_val_rmse = float("inf")
    best_params = None
    best_model = None
    
    np.random.seed(config["project"]["seed"])
    
    for trial in range(n_trials):
        params = {
            "max_depth": int(np.random.uniform(*param_space["max_depth"])),
            "learning_rate": np.random.uniform(*param_space["learning_rate"]),
            "n_estimators": int(np.random.uniform(*param_space["n_estimators"])),
            "subsample": np.random.uniform(*param_space["subsample"]),
            "colsample_bytree": np.random.uniform(*param_space["colsample_bytree"]),
            "random_state": config["project"]["seed"],
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_params = params
            best_model = model
    
    test_pred = best_model.predict(X_test)
    
    metrics = {
        "train_rmse": np.sqrt(mean_squared_error(y_train, best_model.predict(X_train))),
        "val_rmse": best_val_rmse,
        "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
        "test_mae": mean_absolute_error(y_test, test_pred),
        "test_r2": r2_score(y_test, test_pred),
    }
    
    return {
        "model": "xgboost",
        "metrics": metrics,
        "predictions": test_pred,
        "model_obj": best_model,
        "params": best_params,
    }


def train_lightgbm(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, config: Dict) -> Dict:
    """Train LightGBM."""
    X_train, y_train = prepare_features(train)
    X_val, y_val = prepare_features(val)
    X_test, y_test = prepare_features(test)
    
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        random_state=config["project"]["seed"],
        verbose=-1,
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="rmse", verbose=False)
    
    test_pred = model.predict(X_test)
    
    metrics = {
        "train_rmse": np.sqrt(mean_squared_error(y_train, model.predict(X_train))),
        "val_rmse": np.sqrt(mean_squared_error(y_val, model.predict(X_val))),
        "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
        "test_mae": mean_absolute_error(y_test, test_pred),
        "test_r2": r2_score(y_test, test_pred),
    }
    
    return {
        "model": "lightgbm",
        "metrics": metrics,
        "predictions": test_pred,
        "model_obj": model,
    }


def train_mlp(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, config: Dict) -> Dict:
    """Train tiny MLP."""
    X_train, y_train = prepare_features(train)
    X_val, y_val = prepare_features(val)
    X_test, y_test = prepare_features(test)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train.values)
    y_train_t = torch.FloatTensor(y_train.values).reshape(-1, 1)
    X_val_t = torch.FloatTensor(X_val.values)
    y_val_t = torch.FloatTensor(y_val.values).reshape(-1, 1)
    X_test_t = torch.FloatTensor(X_test.values)
    y_test_t = torch.FloatTensor(y_test.values).reshape(-1, 1)
    
    model = TinyMLP(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    n_epochs = 100
    best_val_rmse = float("inf")
    best_model_state = None
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_rmse = np.sqrt(criterion(val_pred, y_val_t).item())
            
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    model.eval()
    
    with torch.no_grad():
        test_pred = model(X_test_t).numpy().flatten()
    
    metrics = {
        "train_rmse": np.sqrt(mean_squared_error(y_train, model(X_train_t).numpy().flatten())),
        "val_rmse": best_val_rmse,
        "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
        "test_mae": mean_absolute_error(y_test, test_pred),
        "test_r2": r2_score(y_test, test_pred),
    }
    
    return {
        "model": "mlp",
        "metrics": metrics,
        "predictions": test_pred,
        "model_obj": model,
    }


def ensemble_models(results: Dict[str, Dict], val: pd.DataFrame, test: pd.DataFrame) -> Dict:
    """Create ensemble by optimizing weights on validation RMSE."""
    # Get validation predictions
    val_preds = {}
    test_preds = {}
    
    for name, result in results.items():
        if name in ["class_mean", "linear"]:
            continue  # Skip baselines for ensemble
        if "val_predictions" in result:
            val_preds[name] = result["val_predictions"]
        if "predictions" in result:
            test_preds[name] = result["predictions"]
    
    if not val_preds or not test_preds or len(val_preds) < 2:
        return None
    
    # Optimize weights
    from scipy.optimize import minimize
    
    def objective(weights):
        ensemble_val = sum(w * val_preds[name] for name, w in zip(val_preds.keys(), weights))
        val_rmse = np.sqrt(mean_squared_error(val["price_drop_pct"], ensemble_val))
        return val_rmse
    
    # Constraint: weights sum to 1
    constraints = {"type": "eq", "fun": lambda w: sum(w) - 1}
    bounds = [(0, 1) for _ in val_preds]
    initial_weights = [1.0 / len(val_preds)] * len(val_preds)
    
    result = minimize(objective, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    best_weights = result.x
    
    # Apply to test
    ensemble_test = sum(w * test_preds[name] for name, w in zip(test_preds.keys(), best_weights))
    
    metrics = {
        "test_rmse": np.sqrt(mean_squared_error(test["price_drop_pct"], ensemble_test)),
        "test_mae": mean_absolute_error(test["price_drop_pct"], ensemble_test),
        "test_r2": r2_score(test["price_drop_pct"], ensemble_test),
        "weights": dict(zip(test_preds.keys(), best_weights)),
    }
    
    return {"model": "ensemble", "metrics": metrics, "predictions": ensemble_test}


def main():
    """Train all models and log to MLflow."""
    config = load_config()
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(config["logging"]["mlflow_tracking_uri"])
    mlflow.set_experiment("rx_price_erosion")
    
    # Load data
    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    train = pd.read_parquet(artifacts_dir / "train.parquet")
    val = pd.read_parquet(artifacts_dir / "val.parquet")
    test = pd.read_parquet(artifacts_dir / "test.parquet")
    
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    results = {}
    
    # Baselines
    print("Training class mean baseline...")
    results["class_mean"] = baseline_class_mean(train, val, test)
    
    print("Training linear baseline...")
    results["linear"] = baseline_linear(train, val, test)
    
    # Models
    print("Training XGBoost...")
    results["xgboost"] = train_xgboost(train, val, test, config)
    
    print("Training LightGBM...")
    results["lightgbm"] = train_lightgbm(train, val, test, config)
    
    print("Training MLP...")
    results["mlp"] = train_mlp(train, val, test, config)
    
    # Ensemble
    print("Creating ensemble...")
    # Need to add val predictions to results
    X_val, y_val = prepare_features(val)
    if "model_obj" in results["xgboost"] and results["xgboost"]["model_obj"] is not None:
        results["xgboost"]["val_predictions"] = results["xgboost"]["model_obj"].predict(X_val)
    if "model_obj" in results["lightgbm"] and results["lightgbm"]["model_obj"] is not None:
        results["lightgbm"]["val_predictions"] = results["lightgbm"]["model_obj"].predict(X_val)
    # MLP val predictions would need to be computed similarly (requires tensor conversion)
    
    ensemble_result = ensemble_models(results, val, test)
    if ensemble_result:
        results["ensemble"] = ensemble_result
    
    # Log to MLflow
    for name, result in results.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params(result.get("params", {}))
            mlflow.log_metrics(result["metrics"])
            
            if "model_obj" in result and result["model_obj"] is not None:
                if name in ["xgboost", "lightgbm"]:
                    mlflow.sklearn.log_model(result["model_obj"], "model")
    
    # Save metrics
    import json
    metrics_summary = {name: r["metrics"] for name, r in results.items()}
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    with open(reports_dir / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    
    print("Training complete. Metrics saved to reports/metrics.json")


if __name__ == "__main__":
    main()

