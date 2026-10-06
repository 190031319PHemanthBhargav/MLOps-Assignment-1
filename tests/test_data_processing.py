import os
import pandas as pd
import pytest

datapath = "data/heart_disease_cleaned.csv"

@pytest.fixture
def loaded_dataset():
    """Fixture to load the cleaned dataset"""
    assert os.path.exists(datapath), f"Dataset not found at {datapath}"
    return pd.read_csv(datapath)

def test_dataset_not_empty(loaded_dataset):
    """Verify that the dataset is loaded and not empty"""
    assert not loaded_dataset.empty, "The dataset is empty."
    assert loaded_dataset.shape[0] > 0, "The dataset has no rows. Dataset should contain rows."

def test_no_missing_values(loaded_dataset):
    """Verify that there are not missing values in the dataset."""
    null_count = loaded_dataset.isnull().sum().sum()
    assert null_count == 0, f"The dataset contains {null_count} missing values. Dataset should not contain missing values."

def test_target_column_binary(loaded_dataset):
    """Verify target columns contain strictly binary values (0 and 1)""" 
    assert "target" in loaded_dataset.columns, "target column missing from dataset"
    unique_targets = set(loaded_dataset["target"].unique())
    assert unique_targets.issubset({0, 1}), f"Target column contains non-binary values: {unique_targets}. Target column should only contain binary values (0 and 1)."

def test_expected_feature_columns(loaded_dataset):
    """Verify the dataset contains the expected feature columns"""
    expected_columns = {
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    }
    assert expected_columns.issubset(set(loaded_dataset.columns)), "Dataset schema missing required features"