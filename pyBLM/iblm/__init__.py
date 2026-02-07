"""
pyBLM - Interpretable Boosted Linear Models for Python

A Python implementation of Interpretable Boosted Linear Models (IBLM),
combining Generalized Linear Models with XGBoost for enhanced interpretability.

Author: Based on the R package by Karol Gawlowski and Paul Beard
License: MIT
"""

from .models import IBLMModel
from .training import train_iblm_xgb
from .preprocessing import split_into_train_validate_test
from .explanation import explain_iblm
from .predict import predict
from .data import load_fremtpl_mini
from .visualization import (
    plot_glm_residuals,
    plot_feature_importance,
    plot_predictions_vs_actual,
    plot_booster_predictions_distribution,
    plot_model_comparison,
    theme_iblm
)
from .utils import (
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    calculate_pinball_scores,
    model_summary,
    get_glm_coefficients,
    get_residuals,
    check_iblm_model,
    detect_outliers
)

__version__ = "0.1.0"
__author__ = "Zhuowen Zhou (Python implementation based on Karol Gawlowski and Paul Beard's R package)"
__title__ = "pyBLM"
__license__ = "MIT"

__all__ = [
    "IBLMModel",
    "train_iblm_xgb",
    "predict",
    "split_into_train_validate_test",
    "explain_iblm",
    "load_fremtpl_mini",
    "plot_glm_residuals",
    "plot_feature_importance",
    "plot_predictions_vs_actual",
    "plot_booster_predictions_distribution",
    "plot_model_comparison",
    "theme_iblm",
    "calculate_mae",
    "calculate_rmse",
    "calculate_mape",
    "calculate_pinball_scores",
    "model_summary",
    "get_glm_coefficients",
    "get_residuals",
    "check_iblm_model",
    "detect_outliers",
]
