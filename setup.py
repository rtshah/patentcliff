"""Setup script for the project."""
from setuptools import setup, find_packages

setup(
    name="rx-price-erosion",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyarrow>=12.0.0",
        "requests>=2.31.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "torch>=2.0.0",
        "mlflow>=2.8.0",
        "shap>=0.42.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "python-dateutil>=2.8.0",
        "scipy>=1.11.0",
    ],
    python_requires=">=3.9",
)

