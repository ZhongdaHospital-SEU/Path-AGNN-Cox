from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="path-agnn-cox",
    version="0.1.0",
    description="Pathway-constrained adaptive graph neural network for interpretable survival analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["path_agnn_cox*", "baselines*", "benchmark*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24", "pandas>=2.0", "scipy>=1.10", "scikit-learn>=1.3",
        "lifelines>=0.28", "scikit-survival>=0.22", "statsmodels>=0.14",
        "matplotlib>=3.7", "torch>=2.0", "pyyaml>=6.0",
    ],
)