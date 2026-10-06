"""Train, evaluate, and package the selected heart-disease classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "heart_disease_cleaned.csv"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "ca", "thal"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
TARGET_COLUMN = "target"
RANDOM_STATE = 42

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the cleaned CSV dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR,
        help="Directory for model.pkl, metadata, and CV results.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=(ROOT_DIR / "mlruns").as_uri(),
        help="MLflow tracking URI. Defaults to this repository's mlruns directory.",
    )
    parser.add_argument(
        "--experiment-name",
        default="Heart Disease Prediction",
        help="MLflow experiment name.",
    )
    return parser.parse_args()


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("binary", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
    )


def build_classifier(model_name: str, params: dict[str, Any], *, final: bool = False):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **params)
    if model_name == "Random Forest":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if model_name == "RBF SVM":
        return SVC(kernel="rbf", probability=final, random_state=RANDOM_STATE, **params)
    if model_name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=RANDOM_STATE, **params)
    raise ValueError(f"Unsupported model: {model_name}")


def candidate_configurations() -> list[tuple[str, dict[str, Any]]]:
    configurations = [
        ("Logistic Regression", {"C": value})
        for value in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    ]
    configurations.extend(
        ("Random Forest", params)
        for params in ParameterGrid({
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 8],
        })
    )
    configurations.extend(
        ("RBF SVM", params)
        for params in ParameterGrid({
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale", 0.01, 0.1],
        })
    )
    configurations.extend(
        ("Gradient Boosting", params)
        for params in ParameterGrid({
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.03, 0.05, 0.1, 0.2],
            "max_depth": [1, 2],
        })
    )
    return configurations


def summarize_cv_scores(cv_scores: dict[str, np.ndarray]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for metric_name in SCORING:
        fold_scores = cv_scores[f"test_{metric_name}"]
        metrics[f"cv_mean_{metric_name}"] = float(fold_scores.mean())
        metrics[f"cv_std_{metric_name}"] = float(fold_scores.std(ddof=1))
    return metrics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    data_path = args.data_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    # Exclude the original multiclass label and binary target from model inputs.
    X = data.loc[:, FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    preprocessor = build_preprocessor()
    results: list[dict[str, Any]] = []
    configurations = candidate_configurations()

    for experiment_number, (model_name, params) in enumerate(configurations, start=1):
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", build_classifier(model_name, params)),
        ])
        parameter_label = ", ".join(
            f"{name}={value}" for name, value in params.items()
        )
        run_name = f"Tuning {experiment_number:02d} - {model_name} ({parameter_label})"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model_type": model_name,
                "cv_folds": 5,
                "selection_metric": "roc_auc",
                **params,
            })
            cv_scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=SCORING,
                n_jobs=1,
                error_score="raise",
            )
            cv_metrics = summarize_cv_scores(cv_scores)
            mlflow.log_metrics(cv_metrics)

        results.append({
            "model_name": model_name,
            "params": params,
            **cv_metrics,
        })
        print(
            f"[{experiment_number:02d}/{len(configurations)}] {run_name}: "
            f"CV ROC-AUC={cv_metrics['cv_mean_roc_auc']:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values(
        "cv_mean_roc_auc", ascending=False
    ).reset_index(drop=True)
    comparison_path = output_dir / "cv_model_comparison.csv"
    results_df.to_csv(comparison_path, index=False)

    best_result = results_df.iloc[0]
    best_model_name = str(best_result["model_name"])
    best_params = dict(best_result["params"])
    final_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", build_classifier(best_model_name, best_params, final=True)),
    ])
    final_pipeline.fit(X_train, y_train)
    y_pred = final_pipeline.predict(X_test)
    y_proba = final_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_recall": float(recall_score(y_test, y_pred)),
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    model_path = output_dir / "model.pkl"
    joblib.dump(final_pipeline, model_path)
    metadata = {
        "model_name": best_model_name,
        "parameters": best_params,
        "selection_metric": "cv_mean_roc_auc",
        "cv_metrics": {
            metric_name: float(best_result[metric_name])
            for metric_name in results_df.columns
            if metric_name.startswith("cv_")
        },
        "test_metrics": test_metrics,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "excluded_columns": ["num", TARGET_COLUMN],
        "data_sha256": sha256_file(data_path),
        "model_sha256": sha256_file(model_path),
        "random_state": RANDOM_STATE,
        "cv_folds": 5,
        "python_version": __import__("platform").python_version(),
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "joblib_version": joblib.__version__,
        "mlflow_version": mlflow.__version__,
    }
    metadata_path = output_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with mlflow.start_run(run_name=f"Final - {best_model_name}"):
        mlflow.log_params({
            "selected_model": best_model_name,
            "selection_metric": "cv_mean_roc_auc",
            **best_params,
        })
        mlflow.log_metrics({
            "best_cv_mean_roc_auc": float(best_result["cv_mean_roc_auc"]),
            **test_metrics,
        })
        mlflow.sklearn.log_model(final_pipeline, artifact_path="model")
        mlflow.log_artifact(str(comparison_path))
        mlflow.log_artifact(str(metadata_path))

    print("\nCross-validation comparison (sorted by mean ROC-AUC):")
    print(results_df.to_string(index=False))
    print(f"\nSelected model: {best_model_name} {best_params}")
    print(f"Model package: {model_path}")
    print(f"Model metadata: {metadata_path}")
    print(f"CV comparison: {comparison_path}")
    print(f"Final test metrics: {test_metrics}")

    subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / "verify_packaging.py"),
            "--model-path",
            str(model_path),
            "--metadata-path",
            str(metadata_path),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
