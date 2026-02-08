"""
Preprocessing utilities for IBLM.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def split_into_train_validate_test(
    df: pd.DataFrame,
    train_prop: float = 0.7,
    validate_prop: float = 0.15,
    test_prop: float = 0.15,
    seed: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Split a DataFrame into train, validate, and test sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data frame to be split.
    train_prop : float, default=0.7
        Proportion of data allocated to training set.
    validate_prop : float, default=0.15
        Proportion of data allocated to validation set.
    test_prop : float, default=0.15
        Proportion of data allocated to test set.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    dict
        Dictionary with keys 'train', 'validate', 'test' containing the respective DataFrames.
    
    Examples
    --------
    >>> from iblm import load_fremtpl_mini, split_into_train_validate_test
    >>> df = load_fremtpl_mini()
    >>> split_data = split_into_train_validate_test(df, seed=9000)
    >>> split_data['train'].shape
    (5000, 11)
    """
    
    # Validate inputs
    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    assert np.isclose(train_prop + validate_prop + test_prop, 1.0), \
        "Proportions must sum to 1.0"
    
    if seed is not None:
        np.random.seed(seed)
    
    # Assign each row to a split
    splits = np.random.choice(
        ['train', 'validate', 'test'],
        size=len(df),
        p=[train_prop, validate_prop, test_prop]
    )
    
    # Create dictionary of splits
    split_dfs = {
        'train': df[splits == 'train'].reset_index(drop=True),
        'validate': df[splits == 'validate'].reset_index(drop=True),
        'test': df[splits == 'test'].reset_index(drop=True),
    }
    
    return split_dfs


def check_required_names(data: dict, required_names: list) -> bool:
    """
    Check that all required names are present in a dictionary.
    
    Parameters
    ----------
    data : dict
        Dictionary to check (e.g., a DataFrame is also a dict-like object).
    required_names : list
        List of required keys.
    
    Returns
    -------
    bool
        True if all required names are present.
    
    Raises
    ------
    ValueError
        If any required names are missing.
    """
    
    # Accept either a dict-like object or a pandas DataFrame
    if isinstance(data, dict):
        keys = set(data.keys())
    elif isinstance(data, pd.DataFrame):
        keys = set(data.columns)
    else:
        raise TypeError("Input must be a dictionary or pandas DataFrame.")

    missing = set(required_names) - keys
    
    if missing:
        raise ValueError(f"Missing required names: {missing}")
    
    return True


def check_data_variability(data: pd.DataFrame, response_var: str) -> bool:
    """
    Check that response and predictor variables have more than one unique value.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data frame containing variables to check.
    response_var : str
        Name of the response variable in `data`.
    
    Returns
    -------
    bool
        True if all checks pass.
    
    Raises
    ------
    ValueError
        If any variable has only one unique value.
    """
    
    # Check response variable
    unique_resp_vals = data[response_var].nunique()
    if unique_resp_vals <= 1:
        raise ValueError(
            f"Response variable '{response_var}' must have more than one unique value. "
            f"Found {unique_resp_vals} unique value(s)."
        )
    
    # Check predictor variables
    unvaried_fields = [col for col in data.columns 
                       if col != response_var and data[col].nunique() <= 1]
    
    if unvaried_fields:
        raise ValueError(
            f"Predictor variables must have more than one unique value. "
            f"The following variables have only one unique value: {unvaried_fields}"
        )
    
    return True


def check_no_missing_values(data: pd.DataFrame) -> bool:
    """
    Check that data contains no missing values.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data frame to check.
    
    Returns
    -------
    bool
        True if no missing values exist.
    
    Raises
    ------
    ValueError
        If any missing values are found.
    """
    
    if data.isnull().any().any():
        missing_cols = data.columns[data.isnull().any()].tolist()
        raise ValueError(
            f"Data contains missing values in columns: {missing_cols}"
        )
    
    return True


def check_no_character_columns(data: pd.DataFrame) -> bool:
    """
    Check that data contains no character/string columns.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data frame to check.
    
    Returns
    -------
    bool
        True if no string columns exist.
    
    Raises
    ------
    ValueError
        If any string columns are found.
    """
    
    char_cols = data.select_dtypes(include=['object', 'string']).columns.tolist()
    
    if char_cols:
        raise ValueError(
            f"Data cannot contain string/character columns. Convert to categorical. "
            f"Found: {char_cols}"
        )
    
    return True


def identify_variable_types(data: pd.DataFrame, response_var: str) -> Tuple[list, list]:
    """
    Identify continuous and categorical variables in data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data frame to analyze.
    response_var : str
        Name of response variable to exclude.
    
    Returns
    -------
    tuple
        (continuous_vars, categorical_vars) - lists of variable names
    """
    
    predictors = [col for col in data.columns if col != response_var]
    
    continuous = []
    categorical = []
    
    for col in predictors:
        if pd.api.types.is_categorical_dtype(data[col]):
            categorical.append(col)
        elif pd.api.types.is_numeric_dtype(data[col]):
            # Check if it's actually categorical by number of unique values
            # Treat binary or near-binary numeric variables as categorical
            if data[col].nunique() <= 2:
                categorical.append(col)
            else:
                continuous.append(col)
        else:
            categorical.append(col)
    
    return continuous, categorical
