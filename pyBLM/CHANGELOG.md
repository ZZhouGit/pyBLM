# Changelog

## [Unreleased]

## [0.1.0] - 2024-02-07

### Added
- Initial release of pyBLM
- Python translation of IBLM R package
- Core IBLM model (GLM + XGBoost)
- Support for Poisson, Gamma, Tweedie, Gaussian families
- Prediction with booster trimming
- SHAP-based model explanation
- Data preprocessing and visualization utilities
- Hyperparameter tuning utilities
- Cross-validation support
- GPU acceleration
- Sphinx documentation site
- More visualization types
- Performance benchmarking tools

### Known Limitations
- Currently only supports XGBoost as booster model
- Categorical handling could be more sophisticated
- Limited cross-validation utilities
- No hyperparameter optimization built-in

## Comparison with R Package

The Python version (pyBLM) aims for feature parity with the R package (IBLM v1.0.2). Key differences:
- Uses statsmodels GLM instead of base R GLM
- Uses Python's SHAP library for explanations (compatible with R's approach)
- Matplotlib/seaborn for visualization instead of ggplot2
- xgboost API is similar in both versions
