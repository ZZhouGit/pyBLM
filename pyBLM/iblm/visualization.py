"""
Visualization utilities for IBLM models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple
from iblm.models import IBLMModel


def plot_glm_residuals(
    iblm_model: IBLMModel,
    data_split: str = 'train',
    figsize: Tuple[int, int] = (12, 4)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create diagnostic plots for GLM residuals.
    
    Parameters
    ----------
    iblm_model : IBLMModel
        Fitted IBLM model.
    data_split : str, default='train'
        Which data split to use: 'train', 'validate', or 'test'.
    figsize : tuple, default=(12, 4)
        Figure size.
    
    Returns
    -------
    fig, axes
        Matplotlib figure and axes objects.
    """
    
    if data_split not in iblm_model.data:
        raise ValueError(f"Data split '{data_split}' not available")
    
    data = iblm_model.data[data_split]
    responses = data['responses'].values
    glm_preds = data['glm_preds']
    
    # Calculate residuals based on relationship
    if iblm_model.relationship == 'multiplicative':
        residuals = responses / glm_preds
    else:
        residuals = responses - glm_preds
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Residuals vs Fitted
    axes[0].scatter(glm_preds, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Fitted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Fitted Values')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Histogram of residuals
    axes[2].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Residuals')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of Residuals')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes


def plot_feature_importance(
    iblm_model: IBLMModel,
    top_n: int = 15,
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot feature importance from the booster model.
    
    Parameters
    ----------
    iblm_model : IBLMModel
        Fitted IBLM model.
    top_n : int, default=15
        Number of top features to display.
    figsize : tuple, default=(10, 6)
        Figure size.
    
    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects.
    """
    
    import xgboost as xgb
    
    # Get feature importance
    importance = iblm_model.booster_model.get_score(importance_type='weight')
    
    # Convert to DataFrame and sort
    importance_df = pd.DataFrame(
        list(importance.items()),
        columns=['Feature', 'Importance']
    ).sort_values('Importance', ascending=True).tail(top_n)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    importance_df.plot(x='Feature', y='Importance', kind='barh', ax=ax, legend=False)
    ax.set_title(f'Top {top_n} Feature Importance (Booster Model)')
    ax.set_xlabel('Importance (Number of times used)')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    return fig, ax


def plot_predictions_vs_actual(
    iblm_model: IBLMModel,
    predictions: np.ndarray,
    data_split: str = 'test',
    figsize: Tuple[int, int] = (8, 6)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create scatter plot of predictions vs actual values.
    
    Parameters
    ----------
    iblm_model : IBLMModel
        Fitted IBLM model.
    predictions : np.ndarray
        Model predictions.
    data_split : str, default='test'
        Which data split was used for predictions.
    figsize : tuple, default=(8, 6)
        Figure size.
    
    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects.
    """
    
    if data_split not in iblm_model.data:
        raise ValueError(f"Data split '{data_split}' not available")
    
    actual = iblm_model.data[data_split]['responses'].values
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(actual, predictions, alpha=0.5)
    
    # Add diagonal line
    min_val = min(actual.min(), predictions.min())
    max_val = max(actual.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
    
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'Predictions vs Actual ({data_split.title()} Data)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig, ax


def plot_booster_predictions_distribution(
    iblm_model: IBLMModel,
    booster_preds: np.ndarray,
    figsize: Tuple[int, int] = (10, 4)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create diagnostic plots for booster predictions.
    
    Parameters
    ----------
    iblm_model : IBLMModel
        Fitted IBLM model.
    booster_preds : np.ndarray
        Booster model predictions.
    figsize : tuple, default=(10, 4)
        Figure size.
    
    Returns
    -------
    fig, axes
        Matplotlib figure and axes objects.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    axes[0].hist(booster_preds, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(booster_preds), color='r', linestyle='--', label=f'Mean: {np.mean(booster_preds):.3f}')
    axes[0].set_xlabel('Booster Predictions')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Booster Predictions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Density plot
    pd.Series(booster_preds).plot(kind='density', ax=axes[1])
    axes[1].set_xlabel('Booster Predictions')
    axes[1].set_title('Kernel Density Estimate of Booster Predictions')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes


def plot_model_comparison(
    iblm_predictions: np.ndarray,
    glm_predictions: np.ndarray,
    actual: np.ndarray,
    figsize: Tuple[int, int] = (12, 5)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Compare predictions from IBLM vs GLM only.
    
    Parameters
    ----------
    iblm_predictions : np.ndarray
        Ensemble model predictions.
    glm_predictions : np.ndarray
        GLM-only predictions.
    actual : np.ndarray
        Actual values.
    figsize : tuple, default=(12, 5)
        Figure size.
    
    Returns
    -------
    fig, axes
        Matplotlib figure and axes objects.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # GLM predictions
    axes[0].scatter(actual, glm_predictions, alpha=0.5, label='GLM')
    min_val = min(actual.min(), glm_predictions.min())
    max_val = max(actual.max(), glm_predictions.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title('GLM Only Predictions')
    axes[0].grid(True, alpha=0.3)
    
    # IBLM predictions
    axes[1].scatter(actual, iblm_predictions, alpha=0.5, label='IBLM', color='green')
    min_val = min(actual.min(), iblm_predictions.min())
    max_val = max(actual.max(), iblm_predictions.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title('IBLM Predictions')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes


def theme_iblm():
    """Apply IBLM theme to matplotlib plots."""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Set color palette
    sns.set_palette("husl")
