from setuptools import setup, find_packages

setup(
    name="beatlabile",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.11",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "vitaldb>=1.4",
        "wfdb>=4.1",
        "statsmodels>=0.14",
        "PuLP>=2.7",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "joblib>=1.3",
        "PyYAML>=6.0",
        "tqdm>=4.66",
        "pyarrow>=14.0",
    ],
)
