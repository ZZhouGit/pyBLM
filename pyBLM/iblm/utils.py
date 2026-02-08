"""
Utility functions for IBLM.
"""

import pandas as pd
import numpy as np
from iblm.models import IBLMModel
from typing import Tuple





def get_glm_coefficients(iblm_model: IBLMModel) -> pd.Series:
    """
    Extract GLM coefficients as a named series.
    
    Parameters
    ----------
    iblm_model : IBLMModel
        Fitted IBLM model.
    
    Returns
    -------
    pd.Series
        Series of GLM coefficients with variable names as index.
    """
    
    return iblm_model.glm_model.params


def calculate_pinball_scores(
    actual: np.ndarray,
    predicted: np.ndarray,
    quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75)
) -> dict:
    """
    Calculate pinball losses for different quantiles.

    Parameters
    ----------
    actual : np.ndarray
        Actual values.
    predicted : np.ndarray
        Predicted values.
    quantiles : tuple, default=(0.25, 0.5, 0.75)
        Quantiles for which to calculate pinball loss.

    Returns
    -------
    dict
        Dictionary with quantile: pinball_loss pairs.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError("`actual` and `predicted` must have the same shape")

    errors = actual - predicted
    scores = {}
    for q in quantiles:
        scores[q] = float(np.mean(np.maximum(q * errors, (q - 1) * errors)))

    return scores


def get_residuals(
    iblm_model: IBLMModel,
    data_split: str = 'train'
) -> np.ndarray:
    """
    Get residuals from the GLM component.
    
    Parameters
    ----------
    iblm_model : IBLMModel
        Fitted IBLM model.
    data_split : str, default='train'
        Which data split to use.
    
    Returns
    -------
    np.ndarray
        Residuals.
    """
    
    if data_split not in iblm_model.data:
        raise ValueError(f"Data split '{data_split}' not available")
    
    data = iblm_model.data[data_split]
    actual = data['responses'].values
    glm_preds = data['glm_preds']
    
    if iblm_model.relationship == 'multiplicative':
        return actual / glm_preds
    else:
        return actual - glm_preds


def check_iblm_model(iblm_model: IBLMModel) -> bool:
    """
    Validate that an object is a properly fitted IBLM model.
    
    Parameters
    ----------
    iblm_model : IBLMModel
        Object to validate.
    
    Returns
    -------
    bool
        True if valid.
    
    Raises
    ------
    TypeError
        If not an IBLM model.
    ValueError
        If the model is improperly configured.
    """
    
    if not isinstance(iblm_model, IBLMModel):
        raise TypeError(f"Expected IBLMModel, got {type(iblm_model)}")
    
    if iblm_model.glm_model is None:
        raise ValueError("IBLM model has no GLM component")
    
    if iblm_model.booster_model is None:
        raise ValueError("IBLM model has no booster component")
    
    if iblm_model.relationship not in ['multiplicative', 'additive']:
        raise ValueError(f"Invalid relationship: {iblm_model.relationship}")
    
    return True


def detect_outliers(
    data: pd.DataFrame,
    method: str = 'iqr',
    threshold: float = 1.5
) -> np.ndarray:
    """
    Detect outliers in the data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to check for outliers.
    method : str, default='iqr'
        Method: 'iqr' for Interquartile Range or 'zscore' for Z-score.
    threshold : float, default=1.5
        Threshold for detection (IQR multiplier or Z-score magnitude).
    
    Returns
    -------
    np.ndarray
        Boolean array indicating outliers.
    """
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((data < (Q1 - threshold * IQR)) | (data > (Q3 + threshold * IQR))).any(axis=1)
    
    elif method == 'zscore':
        from scipy import stats
        outliers = (np.abs(stats.zscore(data.select_dtypes(include=[np.number]))) > threshold).any(axis=1)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return outliers.values
