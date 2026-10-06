"""Verify a saved model package by loading it and predicting one raw patient."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT_DIR / "model.pkl"
DEFAULT_METADATA_PATH = ROOT_DIR / "model_metadata.json"

# Synthetic values; this row is not loaded from or fitted against the dataset.
SAMPLE_PATIENT = {
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 240,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0.0,
    "thal": 3.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the serialized sklearn pipeline.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path to the model metadata JSON.",
    )
    return parser.parse_args()


def verify_runtime_versions(metadata: dict[str, Any]) -> None:
    installed_versions = {
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "joblib_version": joblib.__version__,
    }
    mismatches = {
        name: {"packaged": metadata[name], "installed": installed}
        for name, installed in installed_versions.items()
        if metadata.get(name) != installed
    }
    if mismatches:
        raise RuntimeError(
            "The installed runtime differs from the versions used to package the model: "
            f"{json.dumps(mismatches, indent=2)}. Install the pinned requirements and use Python 3.11.9."
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    model_path = args.model_path.resolve()
    metadata_path = args.metadata_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model package not found: {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    verify_runtime_versions(metadata)
    if sha256_file(model_path) != metadata.get("model_sha256"):
        raise ValueError("The model file checksum does not match the packaged metadata.")

    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError("The model artifact must be a fitted sklearn Pipeline.")
    if "preprocessor" not in pipeline.named_steps or "classifier" not in pipeline.named_steps:
        raise ValueError("The pipeline must contain both preprocessor and classifier steps.")

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    if not isinstance(preprocessor, ColumnTransformer) or not hasattr(preprocessor, "transformers_"):
        raise ValueError("The saved preprocessing ColumnTransformer is missing or unfitted.")
    if not hasattr(classifier, "classes_"):
        raise ValueError("The saved classifier is missing or unfitted.")

    feature_columns = metadata["feature_columns"]
    forbidden_columns = {"num", metadata["target_column"]}
    if forbidden_columns.intersection(feature_columns):
        raise ValueError("The package feature schema contains a target-derived or target column.")
    if set(feature_columns) != set(SAMPLE_PATIENT):
        raise ValueError("The synthetic patient schema does not match the packaged feature schema.")

    raw_patient = pd.DataFrame(
        [[SAMPLE_PATIENT[column] for column in feature_columns]],
        columns=feature_columns,
    )
    prediction = pipeline.predict(raw_patient)
    if not hasattr(pipeline, "predict_proba"):
        raise ValueError("The selected model must support probability prediction.")
    probabilities = pipeline.predict_proba(raw_patient)[0]
    classes = pipeline.named_steps["classifier"].classes_

    print("Packaging verification passed.")
    print(f"Model: {metadata['model_name']}")
    print(f"Input columns: {len(raw_patient.columns)} raw clinical features")
    print(f"Predicted target: {int(prediction[0])}")
    print(
        "Class probabilities: "
        + json.dumps({str(label): float(probability) for label, probability in zip(classes, probabilities)})
    )
    print("No fitting or dataset access was performed by this verification script.")


if __name__ == "__main__":
    main()
