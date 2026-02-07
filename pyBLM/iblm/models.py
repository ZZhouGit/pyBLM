"""
Core IBLM model class.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import GLMEstimator
import xgboost as xgb


@dataclass
class IBLMModel:
    """
    Interpretable Boosted Linear Model
    
    Combines a Generalized Linear Model (GLM) with an XGBoost model
    for improved interpretability with competitive predictive power.
    
    Attributes:
        glm_model: Fitted scikit-learn GLM object
        booster_model: Fitted XGBoost booster object
        data: Dictionary containing train/validate/test data
        relationship: "multiplicative" or "additive" - how to combine GLM and booster
        response_var: Name of the response variable
        predictor_vars: Dictionary with "continuous" and "categorical" predictor names
        cat_levels: Dictionary with categorical levels
        coeff_names: Dictionary with coefficient names
        family: Distribution family ("poisson", "gamma", "tweedie", "gaussian")
    """
    
    glm_model: Any
    booster_model: xgb.Booster
    data: Dict[str, Any]
    relationship: str  # "multiplicative" or "additive"
    response_var: str
    predictor_vars: Dict[str, List[str]]
    cat_levels: Dict[str, Any]
    coeff_names: Dict[str, Any]
    family: str = "poisson"
    
    def __repr__(self) -> str:
        return (
            f"IBLMModel(\n"
            f"  family={self.family},\n"
            f"  relationship={self.relationship},\n"
            f"  response_var={self.response_var},\n"
            f"  n_predictors={len(self.predictor_vars['continuous']) + len(self.predictor_vars['categorical'])}\n"
            f")"
        )
    
    def summary(self) -> str:
        """Print a summary of the IBLM model."""
        summary_text = f"""
IBLM Model Summary
==================
Family: {self.family}
Relationship: {self.relationship}
Response Variable: {self.response_var}

Predictor Variables:
  Continuous: {', '.join(self.predictor_vars['continuous'])}
  Categorical: {', '.join(self.predictor_vars['categorical'])}

Training Data Shape: {self.data['train']['features'].shape}
Validation Data Shape: {self.data['validate']['features'].shape}
Test Data Shape: {self.data['test']['features'].shape if 'test' in self.data else 'Not available'}
"""
        return summary_text
