"""
Training functions for IBLM models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import xgboost as xgb
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson, Gamma, Gaussian, Tweedie
from statsmodels.genmod.families.links import Log, Identity
from iblm.preprocessing import (
    check_required_names, 
    check_data_variability,
    check_no_missing_values,
    check_no_character_columns,
    identify_variable_types
)
from iblm.models import IBLMModel
import warnings


def train_iblm_xgb(
    df_list: Dict[str, pd.DataFrame],
    response_var: str,
    family: str = "poisson",
    params: Optional[Dict[str, Any]] = None,
    nrounds: int = 1000,
    objective: Optional[str] = None,
    custom_metric: Optional[str] = None,
    verbose: int = 0,
    early_stopping_rounds: int = 25,
    maximize: Optional[bool] = None,
    seed: int = 0,
    strip_glm: bool = True,
    **kwargs
) -> IBLMModel:
    """
    Train an Interpretable Boosted Linear Model using XGBoost.
    
    Combines a Generalized Linear Model (GLM) with XGBoost trained on GLM residuals.
    
    Parameters
    ----------
    df_list : dict
        Dictionary with keys 'train', 'validate', 'test' containing DataFrames.
        Obtained from split_into_train_validate_test().
    response_var : str
        Name of the response variable column in the datasets.
    family : str, default='poisson'
        Distribution family: "poisson", "gamma", "tweedie", or "gaussian".
    params : dict, optional
        Additional parameters to pass to xgb.train. The 'objective' parameter
        will be set based on family if not provided.
    nrounds : int, default=1000
        Number of boosting rounds.
    objective : str, optional
        XGBoost objective function. If None, will be selected based on family.
    custom_metric : str, optional
        Custom evaluation metric for XGBoost.
    verbose : int, default=0
        Verbosity level for XGBoost.
    early_stopping_rounds : int, default=25
        Rounds without improvement before early stopping.
    maximize : bool, optional
        Whether to maximize the evaluation metric.
    seed : int, default=0
        Random seed for reproducibility.
    strip_glm : bool, default=True
        Whether to remove unnecessary attributes from GLM object to save memory.
    **kwargs
        Additional arguments passed to GLM or XGBoost.
    
    Returns
    -------
    IBLMModel
        Fitted IBLM model object.
    
    Examples
    --------
    >>> from iblm import load_fremtpl_mini, split_into_train_validate_test, train_iblm_xgb
    >>> df = load_fremtpl_mini()
    >>> split_data = split_into_train_validate_test(df, seed=9000)
    >>> model = train_iblm_xgb(split_data, response_var="ClaimRate", family="poisson")
    """
    
    if params is None:
        params = {}
    
    # ==================== Validation Checks ====================
    check_required_names(df_list, ['train', 'validate'])
    check_required_names(df_list['train'], [response_var])
    check_required_names(df_list['validate'], [response_var])
    
    assert len(df_list['train']) > 0, "Training set cannot be empty"
    assert len(df_list['validate']) > 0, "Validation set cannot be empty"
    assert df_list['train'].columns.equals(df_list['validate'].columns), \
        "Train and validate sets must have the same columns"
    
    # Check for missing values and character columns
    for split_name in ['train', 'validate']:
        check_no_missing_values(df_list[split_name])
        check_no_character_columns(df_list[split_name])
    
    check_data_variability(df_list['train'], response_var)
    
    # ==================== GLM Family Setup ====================
    family = family.lower()
    if family not in ["poisson", "gamma", "tweedie", "gaussian"]:
        raise ValueError(
            f"family must be one of: 'poisson', 'gamma', 'tweedie', 'gaussian'. "
            f"Got: {family}"
        )
    
    # Create GLM family object
    glm_family = _get_glm_family(family)
    link_func = glm_family.link
    
    # ==================== XGBoost Parameter Setup ====================
    xgb_params = _get_xgb_params(family, objective, params)
    
    # ==================== Prepare Data ====================
    # Separate response and features
    train_response = df_list['train'][response_var].values
    validate_response = df_list['validate'][response_var].values
    
    train_features = df_list['train'].drop(columns=[response_var])
    validate_features = df_list['validate'].drop(columns=[response_var])
    
    predictor_vars = train_features.columns.tolist()
    continuous_vars, categorical_vars = identify_variable_types(
        df_list['train'], response_var
    )
    
    # ==================== GLM Fitting ====================
    print(f"Training GLM with {family} family...")
    
    # Create design matrix (convert categoricals to numeric)
    train_features_numeric = _prepare_features_for_glm(train_features)
    
    glm_model = sm.GLM(
        train_response,
        train_features_numeric,
        family=glm_family
    ).fit(disp_smry=False)
    
    # ==================== Prepare XGBoost Targets ====================
    # Get GLM predictions
    train_features_numeric_val = _prepare_features_for_glm(validate_features)
    
    train_glm_preds = glm_model.predict(train_features_numeric)
    validate_glm_preds = glm_model.predict(train_features_numeric_val)
    
    # Compute residuals based on link function
    if link_func.__class__.__name__ == 'Log':
        train_targets = train_response / train_glm_preds
        validate_targets = validate_response / validate_glm_preds
        relationship = "multiplicative"
    elif link_func.__class__.__name__ == 'Identity':
        train_targets = train_response - train_glm_preds
        validate_targets = validate_response - validate_glm_preds
        relationship = "additive"
    else:
        raise ValueError(f"Unsupported link function: {link_func}")
    
    # ==================== XGBoost Training ====================
    print(f"Training XGBoost booster...")
    
    # Create DMatrix
    # Use numeric-encoded features for XGBoost (no pandas categorical dtypes)
    dtrain = xgb.DMatrix(
        train_features_numeric,
        label=train_targets,
        feature_names=predictor_vars
    )

    dvalidate = xgb.DMatrix(
        train_features_numeric_val,
        label=validate_targets,
        feature_names=predictor_vars
    )
    
    # Train booster
    evals = [(dtrain, 'train'), (dvalidate, 'validate')]
    
    booster = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=nrounds,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=(verbose > 0),
        xgb_model=None,
    )
    
    # ==================== Prepare Model Metadata ====================
    
    # Get categorical levels
    cat_levels = {}
    if categorical_vars:
        for cat_var in categorical_vars:
            cat_levels[cat_var] = sorted(df_list['train'][cat_var].unique())
    
    # Get GLM coefficients 
    coeff_names = dict(zip(glm_model.model.exog_names, glm_model.params.index))
    
    # Prepare data dictionary
    data = {
        'train': {
            'features': train_features.reset_index(drop=True),
            'responses': pd.Series(train_response, name=response_var),
            'glm_preds': train_glm_preds,
            'targets': train_targets,
        },
        'validate': {
            'features': validate_features.reset_index(drop=True),
            'responses': pd.Series(validate_response, name=response_var),
            'glm_preds': validate_glm_preds,
            'targets': validate_targets,
        }
    }
    
    if 'test' in df_list:
        test_response = df_list['test'][response_var].values
        test_features = df_list['test'].drop(columns=[response_var])
        test_features_numeric = _prepare_features_for_glm(test_features)
        test_glm_preds = glm_model.predict(test_features_numeric)
        
        data['test'] = {
            'features': test_features.reset_index(drop=True),
            'responses': pd.Series(test_response, name=response_var),
            'glm_preds': test_glm_preds,
        }
    
    # Create IBLM model object
    iblm_model = IBLMModel(
        glm_model=glm_model,
        booster_model=booster,
        data=data,
        relationship=relationship,
        response_var=response_var,
        predictor_vars={
            'continuous': continuous_vars,
            'categorical': categorical_vars,
        },
        cat_levels={'all': cat_levels, 'reference': {}},
        coeff_names=coeff_names,
        family=family,
    )
    
    print("IBLM model training complete!")
    
    return iblm_model


def _get_glm_family(family: str):
    """Get statsmodels family object based on family name."""
    
    if family == "poisson":
        return Poisson(link=Log())
    elif family == "gamma":
        return Gamma(link=Log())
    elif family == "tweedie":
        # Tweedie with variance power 1.5
        return Tweedie(link=Log(), var_power=1.5)
    elif family == "gaussian":
        return Gaussian(link=Identity())
    else:
        raise ValueError(f"Unknown family: {family}")


def _get_xgb_params(family: str, objective: Optional[str], params: Dict) -> Dict:
    """Get default XGBoost parameters based on family."""
    
    xgb_params = params.copy()
    
    if objective is not None:
        xgb_params['objective'] = objective
    else:
        if family == "poisson":
            xgb_params['objective'] = "count:poisson"
        elif family == "gamma":
            xgb_params['objective'] = "reg:gamma"
        elif family == "tweedie":
            xgb_params['objective'] = "reg:tweedie"
            xgb_params['tweedie_variance_power'] = 1.5
        elif family == "gaussian":
            xgb_params['objective'] = "reg:squarederror"
    
    # Set default parameters if not specified
    if 'max_depth' not in xgb_params:
        xgb_params['max_depth'] = 5
    if 'eta' not in xgb_params:
        xgb_params['eta'] = 0.1
    if 'subsample' not in xgb_params:
        xgb_params['subsample'] = 0.8
    
    return xgb_params


def _prepare_features_for_glm(features: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical features to numeric for GLM."""
    
    features_numeric = features.copy()
    
    # Convert categorical columns to numeric codes
    for col in features_numeric.columns:
        if pd.api.types.is_categorical_dtype(features_numeric[col]):
            features_numeric[col] = features_numeric[col].cat.codes
        elif features_numeric[col].dtype == 'object':
            # Convert object columns to categorical first
            features_numeric[col] = pd.Categorical(features_numeric[col]).codes
    
    return features_numeric
