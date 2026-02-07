"""
Setup configuration for pyiBLM package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="pyBLM",
    version="0.1.0",
    author="Karol Gawlowski, Paul Beard",
    author_email="your-email@example.com",
    description="Interpretable Boosted Linear Models - Combining GLMs with XGBoost for Interpretability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/pyBLM",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/pyBLM/issues",
        "Documentation": "https://github.com/your-username/pyBLM#readme",
        "Source Code": "https://github.com/your-username/pyBLM",
        "Original R Package": "https://github.com/IFoA-ADSWP/IBLM",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Office/Business :: Financial",
    ],
    keywords="machine-learning interpretability glm xgboost shap actuarial insurance",
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "xgboost>=1.5.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "statsmodels>=0.13.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "shap>=0.41.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "black>=21.7b0",
            "flake8>=3.9.2",
            "tox>=3.24.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    zip_safe=False,
    include_package_data=True,
)
