# MLOps-Assignment-1
MLOps Assignment 1

## Reproducible Model Packaging

The standalone trainer uses the cleaned dataset at `data/heart_disease_cleaned.csv`, evaluates Logistic Regression, Random Forest, RBF SVM, and Gradient Boosting configurations with stratified 5-fold cross-validation, selects by mean CV ROC-AUC, and logs runs to MLflow.

The fitted scikit-learn `Pipeline` packages the `ColumnTransformer` preprocessing steps and classifier together. Only the 13 clinical predictor columns are used; the original `num` label and `target` are excluded from model inputs. The training/test split is stratified with `random_state=42`.

### Setup and Run

Use CPython 3.11.9, then install the pinned project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Train all configured candidates and create the package:

```powershell
python train.py
```

By default, this writes `model.pkl`, `model_metadata.json`, and `cv_model_comparison.csv` to the repository root and logs the candidate and final runs to `mlruns/`. Before exiting, `train.py` launches the verifier in a fresh Python process to check the saved artifact. Paths and the MLflow experiment can be overridden with `--data-path`, `--output-dir`, `--tracking-uri`, and `--experiment-name`.

Verify the serialized model in a separate process using a synthetic raw patient row:

```powershell
python verify_packaging.py
```

The verifier checks the runtime versions, package checksum, and input schema, then loads the packaged preprocessing-plus-classifier pipeline and performs inference without fitting or reading the training dataset. It can also be rerun independently with the command above. Only load `model.pkl` files from trusted sources because pickle-based formats can execute code when loaded.
