from setuptools import setup, find_packages

setup(
    name="mlops-energy-forecast",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        "xgboost>=2.0.0",
        "mlflow>=2.10.0",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
    ],
    python_requires=">=3.9",
)