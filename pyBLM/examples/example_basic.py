"""
Example usage of the pyBLM package.

This script demonstrates how to:
1. Load and prepare data
2. Train an IBLM model
3. Make predictions
4. Explain model predictions using SHAP values
5. Visualize results
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import IBLM functions
from iblm import (
    load_fremtpl_mini,
    split_into_train_validate_test,
    train_iblm_xgb,
    predict,
    explain_iblm,
    plot_glm_residuals,
    plot_feature_importance,
    plot_predictions_vs_actual,
    model_summary,
    theme_iblm
)


def main():
    """Run the example IBLM workflow."""
    
    print("=" * 70)
    print("pyBLM - Interpretable Boosted Linear Models for Python")
    print("=" * 70)
    
    # ==================== 1. Load Data ====================
    print("\n1. Loading freMTPL mini dataset (real data from CASdatasets)...")
    df = load_fremtpl_mini()
    print(f"   Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"   Columns: {', '.join(df.columns.tolist())}")
    print("\n   First few rows:")
    print(df.head())
    
    # ==================== 2. Split Data ====================
    print("\n2. Splitting data into train/validate/test...")
    split_data = split_into_train_validate_test(
        df,
        train_prop=0.7,
        validate_prop=0.15,
        test_prop=0.15,
        seed=9000
    )
    print(f"   Train:    {split_data['train'].shape[0]} rows")
    print(f"   Validate: {split_data['validate'].shape[0]} rows")
    print(f"   Test:     {split_data['test'].shape[0]} rows")
    
    # ==================== 3. Train IBLM Model ====================
    print("\n3. Training IBLM model...")
    print("   - GLM family: Poisson")
    print("   - Booster: XGBoost")
    print("   - Response: ClaimRate")
    
    iblm_model = train_iblm_xgb(
        split_data,
        response_var="ClaimRate",
        family="poisson",
        nrounds=100,
        early_stopping_rounds=10,
        verbose=1
    )
    
    print(iblm_model.summary())
    
    # ==================== 4. Model Summary ====================
    print("\n4. Model Performance Summary")
    summary = model_summary(iblm_model, data_split='train')
    print(f"   Training set size: {summary['n_observations']}")
    print(f"   GLM MAE:  {summary['glm_mae']:.4f}")
    print(f"   GLM RMSE: {summary['glm_rmse']:.4f}")
    print(f"   GLM MAPE: {summary['glm_mape']:.2f}%")
    
    # ==================== 5. Make Predictions ====================
    print("\n5. Making predictions on test set...")
    test_predictions = predict(iblm_model, split_data['test'])
    print(f"   Generated {len(test_predictions)} predictions")
    print(f"   Prediction range: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]")
    print(f"   Prediction mean:  {test_predictions.mean():.4f}")
    
    # ==================== 6. Evaluate Model ====================
    print("\n6. Model Evaluation on Test Set")
    actual = iblm_model.data['test']['responses'].values
    
    from iblm.utils import calculate_mae, calculate_rmse, calculate_mape
    mae = calculate_mae(actual, test_predictions)
    rmse = calculate_rmse(actual, test_predictions)
    mape = calculate_mape(actual, test_predictions)
    
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    
    # ==================== 7. Explain Predictions ====================
    print("\n7. Generating SHAP-based explanations...")
    explainer = explain_iblm(iblm_model, split_data['test'])
    
    print("   SHAP values shape:", explainer['shap'].shape)
    print("   Beta corrections shape:", explainer['beta_corrections'].shape)
    print("   Data beta coefficients shape:", explainer['data_beta_coeff'].shape)
    
    shap_summary = explainer['shap'].describe()
    print("\n   SHAP value statistics:")
    print(shap_summary)
    
    # ==================== 8. Visualizations ====================
    print("\n8. Creating visualizations...")
    theme_iblm()
    
    # Set up figure
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: GLM Residuals
    print("   - Creating GLM residual diagnostics...")
    fig1, axes1 = plot_glm_residuals(iblm_model, data_split='train')
    
    # Plot 2: Feature Importance
    print("   - Creating feature importance plot...")
    fig2, ax2 = plot_feature_importance(iblm_model, top_n=10)
    
    # Plot 3: Predictions vs Actual
    print("   - Creating predictions vs actual plot...")
    fig3, ax3 = plot_predictions_vs_actual(
        iblm_model,
        test_predictions,
        data_split='test'
    )
    
    # ==================== 9. Summary ====================
    print("\n" + "=" * 70)
    print("IBLM Model Training Complete!")
    print("=" * 70)
    print("\nModel Summary:")
    print(iblm_model)
    
    print("\nKey Outputs:")
    print(f"  - Model object: iblm_model")
    print(f"  - Predictions: test_predictions")
    print(f"  - Explanations: explainer dict with SHAP values")
    print(f"  - Visualizations: matplotlib figures")
    
    print("\nNext Steps:")
    print("  1. Use explainer['shap'] for SHAP-based interpretability")
    print("  2. Call explainer['beta_corrected_scatter'](variable_name) for detailed plots")
    print("  3. Examine beta_corrections for individual feature contributions")
    
    # Show plots
    if sys.platform != 'win32' or True:  # Force show
        plt.show()
    
    return iblm_model, explainer, test_predictions


if __name__ == "__main__":
    iblm_model, explainer, predictions = main()
