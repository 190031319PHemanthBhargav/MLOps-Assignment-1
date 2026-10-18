# Heart Disease Prediction Using MLOps Principles

## Abstract

This project implements an end-to-end machine learning pipeline for heart disease prediction using MLOps practices. The system trains and compares multiple classification models on a cleaned clinical dataset, selects the best-performing model based on cross-validation metrics, packages the trained pipeline for deployment, and exposes it via a FastAPI service. The solution also integrates experiment tracking with MLflow, containerization via Docker, and Kubernetes-based orchestration for deployment in a production-like environment.

The project demonstrates how a machine learning model can be developed, validated, packaged, deployed, and monitored in a reproducible and scalable manner.

## 1. Problem Statement

Heart disease prediction is a critical healthcare task where early diagnosis can support medical decision-making and reduce risk. However, building a reliable predictive system involves not just model selection but also reproducibility, artifact management, validation, deployment, and monitoring.

This assignment addresses the need to implement a complete MLOps workflow for binary classification using clinical health data.

## 2. Objectives

The key objectives of the project are:

- preprocess and validate heart disease clinical data
- train and compare several machine learning classifiers
- select the best model using cross-validation
- log experiments and metrics using MLflow
- serialize the trained model with metadata for traceability
- deploy the model as an inference API using FastAPI
- containerize the API using Docker
- deploy the application in a Kubernetes environment
- validate the project using automated tests

## 3. Dataset

The project uses the cleaned heart disease dataset located at:

- `data/heart_disease_cleaned.csv`

The dataset contains clinical features such as:

- age
- sex
- chest pain type (`cp`)
- resting blood pressure (`trestbps`)
- cholesterol (`chol`)
- fasting blood sugar (`fbs`)
- resting ECG (`restecg`)
- maximum heart rate (`thalach`)
- exercise-induced angina (`exang`)
- oldpeak
- slope
- number of major vessels (`ca`)
- thalassemia category (`thal`)

The target variable is `target`, encoded as binary values:

- 0 = no heart disease
- 1 = heart disease present

## 4. Methodology

### 4.1 Data Processing

The dataset is cleaned and validated before model training. The project checks for:

- missing values
- required feature columns
- valid target encoding
- dataset non-emptiness

Only the feature columns used for modeling are retained, while target-derived variables and the original target label are excluded from the model input schema.

### 4.2 Model Training

The training workflow evaluates multiple classifiers:

- Logistic Regression
- Random Forest
- RBF SVM
- Gradient Boosting

Each model is trained using a preprocessing pipeline with:

- numeric scaling using `StandardScaler`
- one-hot encoding for categorical features
- passthrough handling for binary features

Model comparison is performed using 5-fold stratified cross-validation. The selection metric is mean ROC-AUC.

### 4.3 Model Selection and Evaluation

The best-performing configuration identified by the project is:

- Model: Logistic Regression
- Parameter: `C = 0.5`
- Cross-validation ROC-AUC: `0.9039`
- Test ROC-AUC: `0.9364`

This result is stored in `cv_model_comparison.csv` and the full model metadata is stored in `model_metadata.json`.

## 5. MLOps Workflow Implemented

This project follows a complete MLOps lifecycle:

1. Data preparation and validation
2. Model experimentation and comparison
3. MLflow experiment tracking
4. Artifact packaging and metadata generation
5. Deployment-ready model serialization
6. API deployment for inference
7. Containerization and orchestration
8. Monitoring and testing

## 6. Project Structure

```text
MLOps-Assignment-1/
├── app.py                     # FastAPI prediction API
├── train.py                  # Training, model comparison, MLflow logging, packaging
├── verify_packaging.py       # Model verification and schema validation
├── dockerfile                # Docker container definition
├── requirements.txt          # Dependency list
├── pytest.ini                # Pytest configuration
├── LICENSE                   # MIT license
├── model.pkl                 # Serialized trained pipeline
├── model_metadata.json       # Runtime and packaging metadata
├── cv_model_comparison.csv   # Cross-validation comparison results
├── data/
│   └── heart_disease_cleaned.csv
├── notebooks/
│   ├── 01_data_acquisition_and_eda.ipynb
│   └── 02_feature_engineering_and_modeling.ipynb
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── prometheus.yaml
│   └── grafana.yaml
├── tests/
│   ├── test_data_processing.py
│   └── test_model_pipeline.py
├── visualizations/
│   └── training and model evaluation plots
├── screenshots/
│   └── deployment and monitoring screenshots
├── mlruns/
│   └── MLflow experiment artifacts
├── README.md
└── .gitignore
```

## 7. Technologies Used

- Python 3.11.9
- pandas
- NumPy
- scikit-learn
- MLflow
- FastAPI
- Uvicorn
- Prometheus FastAPI Instrumentator
- Docker
- Kubernetes
- pytest

## 8. Setup and Execution

### 8.1 Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 8.2 Train the Model

```powershell
python train.py
```

This script:

- loads the cleaned dataset
- trains and compares candidate models
- logs results to MLflow
- selects the best model
- saves the packaged pipeline as `model.pkl`
- writes metadata to `model_metadata.json`
- stores comparison metrics in `cv_model_comparison.csv`

### 8.3 Verify the Packaged Model

```powershell
python verify_packaging.py
```

This script confirms:

- environment runtime compatibility
- checksum integrity of the saved artifact
- schema consistency for input features
- successful prediction on a synthetic patient record

## 9. API Deployment

The trained model is served through a FastAPI application defined in `app.py`.

### Run locally

```powershell
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

- `GET /health` — checks service health and model readiness
- `POST /predict` — returns prediction and confidence for a patient record

### Example Request

```json
{
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
  "ca": 0,
  "thal": 3
}
```

### Example Response

```json
{
  "prediction": 1,
  "confidence": 0.8142,
  "risk_label": "High Risk"
}
```

## 10. Docker and Kubernetes Deployment

### Docker

```powershell
docker build -t heart-disease-prediction .
docker run -p 8000:8000 heart-disease-prediction
```

### Kubernetes

```powershell
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

The Kubernetes configuration includes:

- deployment with two replicas
- service exposure on port 8000
- resource requests and limits
- integration placeholders for monitoring with Prometheus and Grafana

## 11. Monitoring and Observability

The project includes monitoring for model-serving infrastructure and API runtime metrics:

- Prometheus endpoint exposure via FastAPI instrumentation
- logs for incoming requests and inference responses
- Grafana-ready monitoring configuration

This aligns with the observability expectations of a production MLOps deployment.

## 12. Testing

The project includes unit tests for data validation and model artifact correctness.

```powershell
pytest -q
```

The tests cover:

- dataset non-empty check
- presence of required columns
- binary target validation
- missing value detection
- existence of the packaged model
- model prediction and probability generation

## 13. Results and Discussion

The model comparison results show that the selected Logistic Regression model performs strongly on this dataset. The best configuration achieved a mean cross-validation ROC-AUC of 0.9039, while the test ROC-AUC reached 0.9364. This indicates that the model generalizes well on unseen data and is suitable for use in an inference service.

The project also demonstrates that model performance can be tracked, reproduced, and packaged systematically using MLflow and metadata-driven verification, which are important principles in MLOps.

## 14. Key Academic / Practical Learnings

This assignment emphasizes the importance of:

- reproducible data pipelines
- model comparison and selection
- production-grade packaging of ML artifacts
- deployment of models as services
- monitoring and maintainability
- validating machine learning artifacts before release
- applying software engineering practices to ML systems

## 15. Conclusion

This project successfully demonstrates an end-to-end MLOps workflow for heart disease prediction. It combines data science, software engineering, model deployment, and operational monitoring into a single, reproducible workflow that reflects real-world machine learning production practices.

The solution is not limited to a static model training script; it also shows how to package, deploy, and monitor models in an environment suitable for academic study and real-world use.

## 16. License

This project is distributed under the MIT License. Refer to the `LICENSE` file for full license details.

