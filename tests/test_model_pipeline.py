import os 
import joblib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from train import build_pipeline

Model_path = Path(__file__).resolve().parents[1] / "model.pkl"

@pytest.fixture
def sample_patient_data():
    """Fixture to provide raw sample patient data for testing"""
    return pd.DataFrame([{
        'age': 57.0,
        'sex': 1.0,
        'cp': 2.0,
        'trestbps': 128.0,
        'chol': 229.0,
        'fbs': 0.0,
        'restecg': 0.0,
        'thalach': 150.0,
        'exang': 0.0,
        'oldpeak': 1.0,
        'slope': 1.0,
        'ca': 1.0,
        'thal': 2.0
    }])

def test_pipeline_builder():
    """Verify build_pipeline returns valid scikit-learn Pipeline instance."""
    pipeline = build_pipeline()
    assert hasattr(pipeline, "fit"), "Pipeline should have a fit method."
    assert hasattr(pipeline, "predict"), "Pipeline should have a predict method."

def test_model_artifacts():
    """Verify serialized model.pkl file exists in root directory."""
    assert os.path.exists(Model_path), f"Model artifact not found at {Model_path}. Run train.py first"

def test_model_inference_and_probabilities(sample_patient_data):
    """Verify model inference returns expected output shape and probability values."""
    # Load trained model pipeline
    pipeline = joblib.load(Model_path)

    # Predict class
    prediction = pipeline.predict(sample_patient_data)
    assert isinstance(prediction, np.ndarray), "Prediction output should be a numpy array."
    assert len(prediction) == 1, "Prediction output should contain a single value for one sample."
    assert prediction[0] in [0, 1], "Prediction should be binary (0 or 1)."

    # Predict probabilities
    probabilities = pipeline.predict_proba(sample_patient_data)
    assert isinstance(probabilities, np.ndarray), "Probabilities output should be a numpy array."
    assert probabilities.shape == (1, 2), "Probabilities output should have shape (1, 2)"
    assert np.isclose(probabilities.sum(), 1.0), "Probabilities should sum to 1."

    confidence = probabilities[0][prediction[0]]
    assert 0.0 <= confidence <= 1.0, "Confidence score should be between 0 and 1."
