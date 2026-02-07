# pyBLM - Interpretable Boosted Linear Models for Python

A Python implementation of **Interpretable Boosted Linear Models (IBLM)**, a hybrid modeling approach that combines the transparency of Generalized Linear Models (GLMs) with the predictive power of gradient boosting (XGBoost).

## Overview

IBLM is a technique that fits two complementary models:

1. **Generalized Linear Model (GLM)** - provides the baseline interpretable predictions
2. **XGBoost Booster** - captures patterns that the GLM misses, trained on residuals

This combination delivers:
- ✅ Interpretable coefficients and SHAP explanations
- ✅ Improved predictive power over GLM alone
- ✅ Model transparency with competitive performance

## Features

- **Multiple distribution families**: Poisson, Gamma, Tweedie, Gaussian
- **SHAP-based explanations**: Understand feature contributions at individual prediction level
- **Comprehensive visualization**: Residual diagnostics, feature importance, prediction comparisons
- **Easy to use API**: Simple train-predict-explain workflow
- **Production-ready**: Efficient implementation using statsmodels and XGBoost

## Installation

### From source

```bash
git clone https://github.com/your-username/pyBLM.git
cd pyBLM
pip install -e .
```

### With development dependencies

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from iblm import (
    load_fremtpl_mini,
    split_into_train_validate_test,
    train_iblm_xgb,
    predict,
    explain_iblm
)

# Load sample data
df = load_fremtpl_mini()

# Split into train/validate/test
split_data = split_into_train_validate_test(df, seed=9000)

# Train IBLM model
model = train_iblm_xgb(
    split_data,
    response_var="ClaimRate",
    family="poisson"
)

# Make predictions
predictions = predict(model, split_data['test'])

# Explain predictions with SHAP values
explainer = explain_iblm(model, split_data['test'])
shap_values = explainer['shap']
```

## Usage Examples

See the [examples/](examples/) directory for detailed examples:

- **[example_basic.py](examples/example_basic.py)** - Basic workflow with training, prediction, and explanation

## API Reference

### Core Functions

#### Training
- `train_iblm_xgb()` - Train an IBLM model
  - Parameters: `family` ("poisson", "gamma", "tweedie", "gaussian"), `nrounds`, `early_stopping_rounds`, etc.

#### Prediction
- `predict()` - Generate predictions from trained model
  - Parameters: `trim` for post-hoc trimming of booster predictions

#### Explanation
- `explain_iblm()` - Generate SHAP-based explanations
  - Returns: Dictionary with SHAP values, beta corrections, and plotting functions

#### Preprocessing
- `split_into_train_validate_test()` - Split data into train/validate/test sets
- `check_data_variability()` - Validate data quality

#### Visualization
- `plot_glm_residuals()` - Diagnostic plots for GLM
- `plot_feature_importance()` - Feature importance from booster
- `plot_predictions_vs_actual()` - Prediction quality assessment
- `plot_model_comparison()` - Compare IBLM vs GLM predictions

#### Metrics
- `calculate_mae()`, `calculate_rmse()`, `calculate_mape()`
- `calculate_pinball_scores()` - Quantile loss metrics
- `model_summary()` - Comprehensive model statistics

## Model Structure

```
IBLMModel
├── glm_model: statsmodels GLM object
├── booster_model: XGBoost Booster object
├── data: Training/validation/test data and predictions
├── relationship: "multiplicative" or "additive"
├── response_var: Name of response variable
├── predictor_vars: Lists of continuous and categorical variables
├── cat_levels: Categorical levels
├── family: Distribution family used
└── coeff_names: Coefficient names
```

## Requirements

- Python >= 3.8
- numpy >= 1.21.0
- pandas >= 1.3.0
- xgboost >= 1.5.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- statsmodels >= 0.13.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- shap >= 0.41.0

## Comparison with Original R Package

This Python version maintains feature parity with the original R package:

| Feature | Python | R |
|---------|--------|---|
| Model training | ✅ `train_iblm_xgb()` | ✅ `train_iblm_xgb()` |
| Prediction | ✅ `predict()` | ✅ `predict.iblm()` |
| SHAP explanation | ✅ `explain_iblm()` | ✅ `explain_iblm()` |
| Visualization | ✅ Multiple functions | ✅ Ggplot2-based |
| Multiple families | ✅ Poisson, Gamma, Tweedie, Gaussian | ✅ Same |

## Testing

Run the test suite:

```bash
pytest tests/test_iblm.py -v
```

## Documentation

- **Theory**: See the vignette in [docs/IBLM_theory.md](docs/)
- **API Docs**: Full API documentation in docstrings
- **Examples**: Practical examples in [examples/](examples/)

## Citation

If you use pyiBLM in your research, please cite the original IBLM paper:

```bibtex
@article{gawlowski2025iblm,
  title={Interpretable Boosted Linear Models},
  author={Gawlowski, K. and Wang, Y.},
  journal={ASTIN Bulletin},
  year={2025}
}
```

And cite this Python implementation:

```bibtex
@misc{pyBLM2024,
  title={pyBLM: Python implementation of Interpretable Boosted Linear Models},
  author={Zhuowen Zhou},
  year={2024},
  url={https://github.com/your-username/pyBLM}
}
```

## License

MIT License - see [LICENSE](LICENSE) file

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Authors

- **Original R Package**: Karol Gawlowski, Paul Beard
- **Python Translation**: Zhuowen Zhou

## References

- Gawlowski, K., & Wang, Y. (2025). [Interpretable Boosted Linear Models](https://ifoa-adswp.github.io/IBLM/)
- R Package: [https://github.com/IFoA-ADSWP/IBLM](https://github.com/IFoA-ADSWP/IBLM)
