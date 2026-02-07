"""
Prediction functions for IBLM models.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from iblm.models import IBLMModel
from typing import Union


def predict(
    iblm_model: IBLMModel,
    newdata: pd.DataFrame,
    trim: float = np.nan,
    type: str = "response"
) -> np.ndarray:
    """
    Generate predictions from an IBLM model.
    
    Combines predictions from the GLM and booster model according to the
    relationship type (multiplicative or additive).
    
    Parameters
    ----------
    iblm_model : IBLMModel
        Fitted IBLM model object from train_iblm_xgb().
    newdata : pd.DataFrame
        Data frame with predictor variables matching training data structure.
    trim : float, optional
        Post-hoc truncation of booster predictions. If NaN (default), no trimming
        is applied. The predicted booster values are clipped to [max(1-trim, 0), 1+trim].
    type : str, default="response"
        Type of prediction. Currently only "response" is supported.
    
    Returns
    -------
    np.ndarray
        Array of ensemble predictions.
    
    Raises
    ------
    ValueError
        If type is not "response" or relationship type is invalid.
    
    Examples
    --------
    >>> from iblm import load_fremtpl_mini, split_into_train_validate_test
    >>> from iblm import train_iblm_xgb, predict
    >>> df = load_fremtpl_mini()
    >>> split_data = split_into_train_validate_test(df, seed=9000)
    >>> model = train_iblm_xgb(split_data, response_var="ClaimRate", family="poisson")
    >>> predictions = predict(model, split_data['test'])
    >>> predictions[:5]
    array([0.15, 0.22, 0.18, ...])
    """
    
    if type != "response":
        raise ValueError(f"Only 'response' type is currently supported. Got: {type}")
    
    # Remove response variable if present
    response_var = iblm_model.response_var
    if response_var in newdata.columns:
        newdata = newdata.drop(columns=[response_var])
    
    # Get GLM predictions
    glm_preds = _predict_glm(iblm_model, newdata, type=type)
    
    # Get booster predictions
    booster_preds = _predict_booster(iblm_model, newdata, type=type)
    
    # Apply trimming if specified
    if not np.isnan(trim):
        booster_preds = _apply_trim(booster_preds, trim)
    
    # Combine GLM and booster predictions
    if iblm_model.relationship == "multiplicative":
        predictions = glm_preds * booster_preds
    elif iblm_model.relationship == "additive":
        predictions = glm_preds + booster_preds
    else:
        raise ValueError(
            f"Invalid relationship: {iblm_model.relationship}. "
            f"Must be 'multiplicative' or 'additive'."
        )
    
    return predictions


def _predict_glm(iblm_model: IBLMModel, newdata: pd.DataFrame, type: str = "response") -> np.ndarray:
    """Generate predictions from the GLM component."""
    
    # Prepare features
    newdata_numeric = _prepare_features_for_prediction(newdata, iblm_model)
    
    # Get GLM predictions
    glm_preds = iblm_model.glm_model.predict(newdata_numeric)
    
    return glm_preds


def _predict_booster(iblm_model: IBLMModel, newdata: pd.DataFrame, type: str = "response") -> np.ndarray:
    """Generate predictions from the booster component."""
    
    # Create DMatrix
    dmatrix = xgb.DMatrix(newdata)
    
    # Get booster predictions
    booster_preds = iblm_model.booster_model.predict(dmatrix)
    
    return booster_preds


def _apply_trim(predictions: np.ndarray, trim: float) -> np.ndarray:
    """Apply post-hoc trimming to booster predictions."""
    
    # Clip predictions
    lower_bound = max(1 - trim, 0)
    upper_bound = 1 + trim
    trimmed = np.clip(predictions, lower_bound, upper_bound)
    
    # Rescale to have mean 1
    mean_trim = np.mean(trimmed)
    trimmed = trimmed / mean_trim
    
    return trimmed


def _prepare_features_for_prediction(newdata: pd.DataFrame, iblm_model: IBLMModel) -> pd.DataFrame:
    """Convert categorical features to numeric for prediction."""
    
    newdata_numeric = newdata.copy()
    
    # Convert categorical columns to numeric codes
    for col in newdata_numeric.columns:
        if pd.api.types.is_categorical_dtype(newdata_numeric[col]):
            newdata_numeric[col] = newdata_numeric[col].cat.codes
        elif newdata_numeric[col].dtype == 'object':
            # Convert object columns to categorical first
            newdata_numeric[col] = pd.Categorical(newdata_numeric[col]).codes
    
    return newdata_numeric
