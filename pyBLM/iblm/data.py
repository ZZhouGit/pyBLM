"""
Data loading utilities for IBLM.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings


def load_fremtpl_mini() -> pd.DataFrame:
    """
    Load the freMTPL mini dataset for examples and testing.
    
    The freMTPL (French Motor Third Party Liability) dataset is a real insurance
    dataset from the CASdatasets repository. This is a mini version (25,000 samples)
    derived from freMTPL2freq with computed claim rates.
    
    Returns
    -------
    pd.DataFrame
        Real insurance claim dataset with ~25,000 rows and 10 columns.
        
        Columns include:
        - DrivAge: Driver age (years)
        - VehAge: Vehicle age (years, capped at 50)
        - VehBrand: Vehicle brand (categorical)
        - VehGas: Fuel type - Diesel or Regular (categorical)
        - VehClass: Vehicle class (categorical)
        - Area: Urban/Rural/Suburban (categorical)
        - ClaimRate: Claim frequency per unit exposure (target variable)
        
    Notes
    -----
    If internet access fails, falls back to synthetic data with warning.
    
    Source: CASdatasets package - freMTPL2freq dataset
    Reference: https://github.com/dutangc/CASdatasets
    
    Examples
    --------
    >>> from iblm import load_fremtpl_mini
    >>> df = load_fremtpl_mini()
    >>> df.shape
    (25000, 10)
    >>> df.head()
    """
    
    try:
        # Try to load real data from CASdatasets GitHub
        commit = "c49cbbb37235fc49616cac8ccac32e1491cdc619"
        url = f"https://github.com/dutangc/CASdatasets/raw/{commit}/data/freMTPL2freq.csv"
        
        df = pd.read_csv(url)
        
        # Transform to match R package processing
        df['ClaimRate'] = df['ClaimNb'] / df['Exposure']
        # Cap claim rate at 99.9th percentile like R package does
        df['ClaimRate'] = df['ClaimRate'].clip(upper=df['ClaimRate'].quantile(0.999))
        # Cap vehicle age at 50
        df['VehAge'] = df['VehAge'].clip(upper=50)
        
        # Convert character/string columns to categorical
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].astype('category')
        
        # Drop unnecessary columns (match R package)
        cols_to_drop = ['IDpol', 'Exposure', 'ClaimNb', 'Density', 'Region']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        # Round ClaimRate to integers and sample 25,000 rows with seed 9000
        df['ClaimRate'] = df['ClaimRate'].round().astype('int64')
        np.random.seed(9000)
        df = df.sample(n=25000, random_state=9000).reset_index(drop=True)
        
        return df
        
    except Exception as e:
        # Fallback to synthetic data if download fails
        warnings.warn(
            f"Could not load real freMTPL data from GitHub ({str(e)}). "
            "Using synthetic data instead. Install via: "
            "pip install requests to enable real data download.",
            UserWarning
        )
        return _generate_synthetic_fremtpl_mini()


def _generate_synthetic_fremtpl_mini() -> pd.DataFrame:
    """
    Generate synthetic freMTPL-like data as fallback.
    
    Used when real data cannot be downloaded from GitHub.
    """
    np.random.seed(9000)
    n_samples = 25000
    
    data = {
        'DrivAge': np.random.normal(45, 15, n_samples).astype(int).clip(18, 90),
        'VehAge': np.random.normal(8, 5, n_samples).astype(int).clip(0, 50),
        'VehBrand': np.random.choice(['A', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6'], n_samples),
        'VehGas': np.random.choice(['Diesel', 'Regular'], n_samples),
        'VehClass': np.random.choice(['H', 'M', 'T'], n_samples),
        'Area': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], n_samples),
        'ClaimRate': np.random.poisson(lam=0.1, size=n_samples).astype(int),
    }
    
    df = pd.DataFrame(data)
    
    # Convert categorical columns
    categorical_cols = ['VehBrand', 'VehGas', 'VehClass', 'Area']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    return df
    """Get the path to the data directory."""
    return Path(__file__).parent / 'data'


def save_dataset(df: pd.DataFrame, filename: str) -> str:
    """
    Save a dataset as CSV.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data frame to save.
    filename : str
        Name of the file (without path).
    
    Returns
    -------
    str
        Path to the saved file.
    """
    
    data_dir = get_data_path()
    data_dir.mkdir(exist_ok=True)
    
    filepath = data_dir / filename
    df.to_csv(filepath, index=False)
    
    return str(filepath)


def load_dataset(filename: str) -> pd.DataFrame:
    """
    Load a saved dataset from CSV.
    
    Parameters
    ----------
    filename : str
        Name of the file (without path).
    
    Returns
    -------
    pd.DataFrame
        Loaded data frame.
    """
    
    filepath = get_data_path() / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    return pd.read_csv(filepath)
