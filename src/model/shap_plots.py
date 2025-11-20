"""Generate SHAP plots for model explainability."""
import shap
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
from pathlib import Path as PathLib

project_root = PathLib(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config
from src.model.train import prepare_features


def generate_shap_plots(model, X_test: pd.DataFrame, test: pd.DataFrame, config=None):
    """Generate SHAP global importance bar plot."""
    if config is None:
        config = load_config()
    
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Global importance bar
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(reports_dir / "shap_global.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved SHAP global plot to {reports_dir / 'shap_global.png'}")


def plot_residuals_by_complexity(test: pd.DataFrame, predictions: pd.Series, config=None):
    """Plot residuals grouped by complexity score."""
    if config is None:
        config = load_config()
    
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    residuals = test["price_drop_pct"] - predictions
    
    plt.figure(figsize=(10, 6))
    test.groupby("complexity_score")["price_drop_pct"].apply(
        lambda x: plt.scatter(x.index, residuals[x.index], alpha=0.5, label=f"Complexity {x.name}")
    )
    plt.xlabel("Complexity Score")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residuals by Complexity Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "residuals_by_complexity.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved residuals plot to {reports_dir / 'residuals_by_complexity.png'}")


def plot_pred_vs_actual(test: pd.DataFrame, predictions: pd.Series, config=None):
    """Plot predicted vs actual."""
    if config is None:
        config = load_config()
    
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(test["price_drop_pct"], predictions, alpha=0.5)
    plt.plot([test["price_drop_pct"].min(), test["price_drop_pct"].max()], 
             [test["price_drop_pct"].min(), test["price_drop_pct"].max()], 
             "r--", label="Perfect prediction")
    plt.xlabel("Actual Price Drop %")
    plt.ylabel("Predicted Price Drop %")
    plt.title("Predicted vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "pred_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved pred vs actual plot to {reports_dir / 'pred_vs_actual.png'}")


if __name__ == "__main__":
    config = load_config()
    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    test = pd.read_parquet(artifacts_dir / "test.parquet")
    X_test, y_test = prepare_features(test)
    
    # Load best model (XGBoost)
    import mlflow
    mlflow.set_tracking_uri(config["logging"]["mlflow_tracking_uri"])
    # Would load model from MLflow here
    
    print("SHAP plots module ready")

