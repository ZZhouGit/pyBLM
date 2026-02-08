"""
Unit tests for pyBLM package.
"""

import pytest
import numpy as np
import pandas as pd
from iblm import (
    load_freMTPL_mini,
    split_into_train_validate_test,
    train_iblm_xgb,
    predict,
    explain_iblm,
)
from iblm.preprocessing import (
    check_required_names,
    check_data_variability,
    check_no_missing_values,
    identify_variable_types,
)
from iblm.utils import (
    check_iblm_model,
)

class TestDataLoading:
    """Test data loading functions."""
    
    def test_load_fremtpl_mini(self):
        """Test loading the freMTPL mini dataset."""
        df = load_freMTPL_mini()
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0
        assert 'ClaimRate' in df.columns
        assert 'DrivAge' in df.columns
    
    def test_fremtpl_data_types(self):
        """Test that freMTPL data has correct types."""
        df = load_freMTPL_mini()
        
        # Check categorical columns
        assert pd.api.types.is_categorical_dtype(df['VehBrand'])
        assert pd.api.types.is_categorical_dtype(df['VehGas'])
        
        # Check numeric columns
        assert pd.api.types.is_numeric_dtype(df['ClaimRate'])
        assert pd.api.types.is_numeric_dtype(df['DrivAge'])


class TestPreprocessing:
    """Test preprocessing functions."""
    
    def test_split_into_train_validate_test(self):
        """Test data splitting."""
        df = pd.DataFrame({
            'x': np.random.normal(0, 1, 1000),
            'y': np.random.poisson(1, 1000)
        })
        
        splits = split_into_train_validate_test(df, seed=42)
        
        assert len(splits) == 3
        assert 'train' in splits
        assert 'validate' in splits
        assert 'test' in splits
        
        total = len(splits['train']) + len(splits['validate']) + len(splits['test'])
        assert total == len(df)
    
    def test_split_proportions(self):
        """Test that split proportions are approximately correct."""
        df = pd.DataFrame({
            'x': np.random.normal(0, 1, 10000),
            'y': np.random.poisson(1, 10000)
        })
        
        splits = split_into_train_validate_test(
            df,
            train_prop=0.7,
            validate_prop=0.15,
            test_prop=0.15,
            seed=42
        )
        
        total = len(df)
        train_pct = len(splits['train']) / total
        validate_pct = len(splits['validate']) / total
        test_pct = len(splits['test']) / total
        
        assert 0.65 < train_pct < 0.75
        assert 0.10 < validate_pct < 0.20
        assert 0.10 < test_pct < 0.20
    
    def test_check_required_names(self):
        """Test required names checking."""
        data = {'a': 1, 'b': 2, 'c': 3}
        
        # Should pass
        assert check_required_names(data, ['a', 'b'])
        assert check_required_names(data, ['a'])
        
        # Should raise
        with pytest.raises(ValueError):
            check_required_names(data, ['a', 'd'])
    
    def test_check_data_variability(self):
        """Test data variability checking."""
        # Good data
        good_df = pd.DataFrame({
            'x': [1, 2, 3, 4],
            'y': [1, 1, 2, 2]
        })
        assert check_data_variability(good_df, 'y')
        
        # Bad data - no variability in response
        bad_df = pd.DataFrame({
            'x': [1, 2, 3, 4],
            'y': [1, 1, 1, 1]
        })
        with pytest.raises(ValueError):
            check_data_variability(bad_df, 'y')
    
    def test_identify_variable_types(self):
        """Test variable type identification."""
        df = pd.DataFrame({
            'age': [25, 35, 45, 55],
            'income': [50000, 60000, 70000, 80000],
            'gender': pd.Categorical(['M', 'F', 'M', 'F']),
            'response': [0, 1, 1, 0]
        })
        
        continuous, categorical = identify_variable_types(df, 'response')
        
        assert 'age' in continuous or 'income' in continuous
        assert 'gender' in categorical


class TestTraining:
    """Test model training."""
    
    @pytest.fixture
    def training_data(self):
        """Create sample training data."""
        df = load_freMTPL_mini()
        return split_into_train_validate_test(df, seed=42)
    
    def test_train_iblm_xgb_poisson(self, training_data):
        """Test training IBLM with Poisson family."""
        model = train_iblm_xgb(
            training_data,
            response_var="ClaimRate",
            family="poisson",
            nrounds=10,
        )
        
        assert model is not None
        assert model.family == "poisson"
        assert model.relationship in ['multiplicative', 'additive']
        assert model.glm_model is not None
        assert model.booster_model is not None
    
    def test_train_iblm_xgb_gaussian(self, training_data):
        """Test training IBLM with Gaussian family."""
        model = train_iblm_xgb(
            training_data,
            response_var="ClaimRate",
            family="gaussian",
            nrounds=10,
        )
        
        assert model is not None
        assert model.family == "gaussian"
    
    def test_iblm_model_structure(self, training_data):
        """Test that IBLM model has correct structure."""
        model = train_iblm_xgb(
            training_data,
            response_var="ClaimRate",
            family="poisson",
            nrounds=10,
        )
        
        assert hasattr(model, 'glm_model')
        assert hasattr(model, 'booster_model')
        assert hasattr(model, 'data')
        assert hasattr(model, 'relationship')
        assert hasattr(model, 'response_var')


class TestPrediction:
    """Test prediction functions."""
    
    @pytest.fixture
    def trained_model(self):
        """Create a trained IBLM model."""
        df = load_freMTPL_mini()
        split_data = split_into_train_validate_test(df, seed=42)
        model = train_iblm_xgb(
            split_data,
            response_var="ClaimRate",
            family="poisson",
            nrounds=10,
        )
        return model
    
    def test_predict_returns_array(self, trained_model):
        """Test that predict returns a numpy array."""
        test_data = trained_model.data['test']['features']
        predictions = predict(trained_model, test_data)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(test_data)
    
    def test_predict_positive_values(self, trained_model):
        """Test that predictions are positive for Poisson model."""
        test_data = trained_model.data['test']['features']
        predictions = predict(trained_model, test_data)
        
        # For Poisson GLM, predictions should be non-negative
        assert np.all(predictions >= 0)
    
    def test_predict_with_trim(self, trained_model):
        """Test prediction with trimming."""
        test_data = trained_model.data['test']['features']
        
        predictions_no_trim = predict(trained_model, test_data)
        predictions_trim = predict(trained_model, test_data, trim=0.1)
        
        assert len(predictions_no_trim) == len(predictions_trim)


class TestExplanation:
    """Test explanation functions."""
    
    @pytest.fixture
    def trained_model_with_test(self):
        """Create a trained IBLM model with test set."""
        df = load_freMTPL_mini()
        split_data = split_into_train_validate_test(df, seed=42)
        model = train_iblm_xgb(
            split_data,
            response_var="ClaimRate",
            family="poisson",
            nrounds=10,
        )
        return model
    
    def test_explain_iblm_returns_dict(self, trained_model_with_test):
        """Test that explain_iblm returns a dictionary."""
        test_data = trained_model_with_test.data['test']['features']
        explainer = explain_iblm(trained_model_with_test, test_data)
        
        assert isinstance(explainer, dict)
        assert 'shap' in explainer
        assert 'beta_corrections' in explainer
        assert 'data_beta_coeff' in explainer
    
    def test_shap_values_shape(self, trained_model_with_test):
        """Test that SHAP values have correct shape."""
        test_data = trained_model_with_test.data['test']['features']
        explainer = explain_iblm(trained_model_with_test, test_data)
        
        shap_df = explainer['shap']
        assert isinstance(shap_df, pd.DataFrame)
        assert len(shap_df) == len(test_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
