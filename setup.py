"""Setup for path-agnn-cox."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="path-agnn-cox",
    version="0.1.2",
    description="Pathway-constrained adaptive graph neural network for interpretable survival analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Zhipeng Wang",
    license="MIT",
    url="https://github.com/wangzhipeng-1/Path-AGNN-Cox",
    project_urls={
        "Source Code": "https://github.com/wangzhipeng-1/Path-AGNN-Cox",
        "Bug Tracker": "https://github.com/wangzhipeng-1/Path-AGNN-Cox/issues",
        "Manuscript": "https://github.com/wangzhipeng-1/Path-AGNN-Cox/tree/main/manuscript",
    },
    packages=find_packages(include=["path_agnn_cox*", "baselines*", "benchmark*"]),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24", "pandas>=2.0", "scipy>=1.10", "scikit-learn>=1.3",
        "lifelines>=0.28", "scikit-survival>=0.22", "statsmodels>=0.14",
        "matplotlib>=3.7", "torch>=2.0", "pyyaml>=6.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords="survival analysis, graph neural network, pathway, cancer, TCGA, GEO, interpretability",
)
