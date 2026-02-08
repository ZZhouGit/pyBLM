"""
Data loading utilities for IBLM.
"""

import pandas as pd
import numpy as np
import tempfile
import requests
import pyreadr


def load_freMTPL2freq() -> pd.DataFrame:
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
    
    # Try to load real data from CASdatasets GitHub
    commit = "c49cbbb37235fc49616cac8ccac32e1491cdc619"
    url = f"https://github.com/dutangc/CASdatasets/raw/{commit}/data/freMTPL2freq.rda"
    
    df = pd.read_csv(url)
    
    # Transform to match R package processing
    df['ClaimRate'] = df['ClaimNb'] / df['Exposure']
    # Cap claim rate at 99.9th percentile <-- kept in to help rec with original paper
    df['ClaimRate'] = df['ClaimRate'].clip(upper=df['ClaimRate'].quantile(0.999))
    # Cap vehicle age at 50 <-- kept in to help rec with original paper
    df['VehAge'] = df['VehAge'].clip(upper=50)
    
    # Convert character/string columns to categorical
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    # Drop unnecessary columns
    cols_to_drop = ['IDpol', 'Exposure']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    return df

def load_freMTPL_mini() -> pd.DataFrame:
    commit = "c49cbbb37235fc49616cac8ccac32e1491cdc619"
    seed_no = 9000

    url = f"https://github.com/dutangc/CASdatasets/raw/{commit}/data/freMTPL2freq.rda"

    # download to a temporary file
    temp = tempfile.NamedTemporaryFile(delete=False)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        temp.write(r.content)
    temp.close()

    # load the .rda file
    result = pyreadr.read_r(temp.name)
    df = result["freMTPL2freq"]

    # ClaimRate = ClaimNb / Exposure
    df["ClaimRate"] = df["ClaimNb"] / df["Exposure"]

    # cap ClaimRate at 99.9th percentile
    cap = df["ClaimRate"].quantile(0.999)
    df["ClaimRate"] = np.minimum(df["ClaimRate"], cap)

    # cap VehAge at 50
    df["VehAge"] = np.minimum(df["VehAge"], 50)

    # convert character columns to categorical
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category")

    # drop columns
    df = df.drop(columns=["IDpol", "Exposure", "ClaimNb", "Density", "Region"])

    # round ClaimRate and convert to integer
    df["ClaimRate"] = df["ClaimRate"].round().astype(int)

    # sample 25,000 rows with fixed seed
    df = df.sample(n=25000, random_state=seed_no)

    # return as a clean pandas DataFrame
    return df
