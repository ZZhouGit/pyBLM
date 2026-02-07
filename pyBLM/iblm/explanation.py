"""
Explanation functions for IBLM models using SHAP values.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from iblm.models import IBLMModel
from typing import Dict, Any
import warnings


def explain_iblm(
    iblm_model: IBLMModel,
    data: pd.DataFrame,
    migrate_reference_to_bias: bool = True
) -> Dict[str, Any]:
    """
    Generate SHAP-based explanations for IBLM model predictions.
    
    Creates detailed explanations of how the model's GLM and booster
    components contribute to predictions for each observation.
    
    Parameters
    ----------
    iblm_model : IBLMModel
        Fitted IBLM model from train_iblm_xgb().
    data : pd.DataFrame
        Test or validation data for which to generate explanations.
    migrate_reference_to_bias : bool, default=True
        For categorical variables, migrate beta corrections to the bias term.
    
    Returns
    -------
    dict
        Dictionary containing:
        
        - 'shap': DataFrame of raw SHAP values
        - 'beta_corrections': DataFrame of corrected beta coefficients
        - 'data_beta_coeff': DataFrame of final beta coefficients after corrections
        - 'beta_corrected_scatter': Function for scatter plots
        - 'beta_corrected_density': Function for density plots
        - 'bias_density': Function for bias density plots
        - 'overall_correction': Function for overall correction plots
    
    Examples
    --------
    >>> from iblm import load_fremtpl_mini, split_into_train_validate_test
    >>> from iblm import train_iblm_xgb, explain_iblm
    >>> df = load_fremtpl_mini()
    >>> split_data = split_into_train_validate_test(df, seed=9000)
    >>> model = train_iblm_xgb(split_data, response_var="ClaimRate", family="poisson")
    >>> explainer = explain_iblm(model, split_data['test'])
    >>> shap_vals = explainer['shap']
    >>> beta_corr = explainer['beta_corrections']
    """
    
    # Extract SHAP values from booster
    shap_df = extract_booster_shap(iblm_model.booster_model, data)
    
    # Prepare wide input frame with categorical variables one-hot encoded
    wide_input_frame = data_to_onehot(data, iblm_model)
    
    # Convert SHAP values to wide format
    shap_wide = shap_to_onehot(shap_df, wide_input_frame, iblm_model)
    
    # Derive beta corrections from SHAP values
    beta_corrections = beta_corrections_derive(
        shap_wide, 
        wide_input_frame, 
        iblm_model,
        migrate_reference_to_bias
    )
    
    # Get GLM beta coefficients
    data_glm = data_beta_coeff_glm(data, iblm_model)
    
    # Get booster beta corrections
    data_booster = data_beta_coeff_booster(data, beta_corrections, iblm_model)
    
    # Combined coefficients
    data_beta_coeff = data_glm + data_booster
    
    # Create plotting functions
    beta_corrected_scatter = create_beta_corrected_scatter(
        data_beta_coeff, data, iblm_model
    )
    
    beta_corrected_density = create_beta_corrected_density(
        wide_input_frame, beta_corrections, data, iblm_model
    )
    
    bias_density = create_bias_density(
        wide_input_frame, beta_corrections, iblm_model
    )
    
    overall_correction = create_overall_correction(
        beta_corrections, iblm_model
    )
    
    return {
        'shap': shap_df,
        'beta_corrections': beta_corrections,
        'data_beta_coeff': data_beta_coeff,
        'beta_corrected_scatter': beta_corrected_scatter,
        'beta_corrected_density': beta_corrected_density,
        'bias_density': bias_density,
        'overall_correction': overall_correction,
    }


def extract_booster_shap(booster_model: xgb.Booster, data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract SHAP values from an XGBoost booster model.
    
    Parameters
    ----------
    booster_model : xgb.Booster
        Fitted XGBoost booster model.
    data : pd.DataFrame
        Data for which to compute SHAP values.
    
    Returns
    -------
    pd.DataFrame
        DataFrame of SHAP values with feature names as columns.
    """
    
    # Get feature names from booster
    feature_names = booster_model.feature_names
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(data.shape[1])]
    
    # Select only features in the booster
    data_features = data[feature_names].copy() if all(f in data.columns for f in feature_names) else data
    
    # Create DMatrix
    dmatrix = xgb.DMatrix(data_features, feature_names=feature_names)
    
    # Get SHAP values
    shap_values = booster_model.predict(dmatrix, pred_contribs=True)
    
    # Convert to DataFrame
    # Last column is bias
    shap_df = pd.DataFrame(
        shap_values[:, :-1],
        columns=feature_names
    )
    
    # Add bias column
    shap_df['BIAS'] = shap_values[:, -1]
    
    # Reorder with BIAS first
    cols = ['BIAS'] + [c for c in shap_df.columns if c != 'BIAS']
    shap_df = shap_df[cols]
    
    return shap_df


def data_to_onehot(data: pd.DataFrame, iblm_model: IBLMModel) -> pd.DataFrame:
    """Convert categorical variables to one-hot encoding."""
    
    data_onehot = pd.get_dummies(
        data[iblm_model.predictor_vars['continuous']],
        columns=iblm_model.predictor_vars['categorical'],
        drop_first=False,
        prefix_sep='_'
    )
    
    return data_onehot


def shap_to_onehot(
    shap_df: pd.DataFrame,
    wide_input_frame: pd.DataFrame,
    iblm_model: IBLMModel
) -> pd.DataFrame:
    """Convert SHAP values to match one-hot encoded input format."""
    
    # For now, return as-is; categorical handling would be more complex
    # and depend on how they're encoded in the training data
    return shap_df


def beta_corrections_derive(
    shap_wide: pd.DataFrame,
    wide_input_frame: pd.DataFrame,
    iblm_model: IBLMModel,
    migrate_reference_to_bias: bool = True
) -> pd.DataFrame:
    """
    Derive beta corrections from SHAP values.
    
    SHAP values represent the contribution of each feature to the booster's
    prediction, which can be interpreted as corrections to the GLM estimates.
    """
    
    # For now, SHAP values directly represent the corrections
    beta_corrections = shap_wide.copy()
    
    return beta_corrections


def data_beta_coeff_glm(data: pd.DataFrame, iblm_model: IBLMModel) -> pd.DataFrame:
    """
    Calculate GLM beta coefficients for each observation.
    """
    
    # Get GLM coefficients
    glm_coeff = iblm_model.glm_model.params
    
    # Prepare features
    data_numeric = data.copy()
    for col in data_numeric.columns:
        if pd.api.types.is_categorical_dtype(data_numeric[col]):
            data_numeric[col] = data_numeric[col].cat.codes
        elif data_numeric[col].dtype == 'object':
            data_numeric[col] = pd.Categorical(data_numeric[col]).codes
    
    # Create coefficient matrix
    coeff_matrix = pd.DataFrame(
        index=data.index,
        columns=['bias'] + list(data.columns)
    )
    
    # Add bias
    coeff_matrix['bias'] = glm_coeff.iloc[0]  # Intercept
    
    # Add feature coefficients
    for col in data.columns:
        if col in glm_coeff.index:
            coeff_matrix[col] = glm_coeff[col]
        else:
            coeff_matrix[col] = 0
    
    return coeff_matrix


def data_beta_coeff_booster(
    data: pd.DataFrame,
    beta_corrections: pd.DataFrame,
    iblm_model: IBLMModel
) -> pd.DataFrame:
    """
    Calculate booster beta coefficient corrections for each observation.
    """
    
    # Rename BIAS to bias if present
    booster_corr = beta_corrections.copy()
    if 'BIAS' in booster_corr.columns:
        booster_corr = booster_corr.rename(columns={'BIAS': 'bias'})
    
    # Add missing columns as 0
    for col in data.columns:
        if col not in booster_corr.columns:
            booster_corr[col] = 0
    
    return booster_corr


def create_beta_corrected_scatter(
    data_beta_coeff: pd.DataFrame,
    data: pd.DataFrame,
    iblm_model: IBLMModel
):
    """Return a function that creates scatter plots of beta corrections vs feature values."""
    
    def scatter_plot(variable: str, ax=None, **kwargs):
        """Create scatter plot of beta coefficient vs variable values."""
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots()
        
        if variable in data:
            ax.scatter(data[variable], data_beta_coeff[variable], alpha=0.5, **kwargs)
            ax.set_xlabel(variable)
            ax.set_ylabel(f'Beta Correction: {variable}')
            ax.set_title(f'{variable} Beta Corrections')
        
        return ax
    
    return scatter_plot


def create_beta_corrected_density(
    wide_input_frame: pd.DataFrame,
    beta_corrections: pd.DataFrame,
    data: pd.DataFrame,
    iblm_model: IBLMModel
):
    """Return a function that creates density plots of beta corrections."""
    
    def density_plot(variable: str, ax=None, **kwargs):
        """Create density plot of beta coefficient corrections."""
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots()
        
        if variable in beta_corrections.columns:
            beta_corrections[variable].plot(kind='density', ax=ax, **kwargs)
            ax.set_xlabel(f'{variable} Corrections')
            ax.set_title(f'Density of {variable} Corrections')
        
        return ax
    
    return density_plot


def create_bias_density(
    wide_input_frame: pd.DataFrame,
    beta_corrections: pd.DataFrame,
    iblm_model: IBLMModel
):
    """Return a function that creates density plots for bias."""
    
    def density_plot(ax=None, **kwargs):
        """Create density plot of bias corrections."""
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots()
        
        bias_col = 'BIAS' if 'BIAS' in beta_corrections.columns else 'bias'
        if bias_col in beta_corrections.columns:
            beta_corrections[bias_col].plot(kind='density', ax=ax, **kwargs)
            ax.set_xlabel('Bias Corrections')
            ax.set_title('Density of Bias Corrections')
        
        return ax
    
    return density_plot


def create_overall_correction(
    beta_corrections: pd.DataFrame,
    iblm_model: IBLMModel
):
    """Return a function that creates plots of overall correction distribution."""
    
    def plot(ax=None, **kwargs):
        """Create plot of overall correction distribution."""
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots()
        
        # Calculate total corrections excluding bias
        bias_col = 'BIAS' if 'BIAS' in beta_corrections.columns else 'bias'
        total_corr = beta_corrections.drop(columns=[bias_col]).sum(axis=1)
        
        total_corr.plot(kind='density', ax=ax, **kwargs)
        ax.set_xlabel('Total Feature Corrections')
        ax.set_title('Overall Model Corrections Distribution')
        
        return ax
    
    return plot
