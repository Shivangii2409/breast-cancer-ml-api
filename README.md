# Breast Cancer Wisconsin Diagnostic ML API

An educational binary-classification project using the Breast Cancer Wisconsin Diagnostic dataset (`data.csv`).

## Model and evaluation
- Model: `StandardScaler` + RBF-kernel Support Vector Classifier (SVC)
- Target: `diagnosis` (`B` = Benign, `M` = Malignant)
- Split: stratified 80/20 holdout, `random_state=42`
- Excludes the identifier (`id`) and empty `Unnamed: 32` column
- `model.joblib` stores the fitted pipeline, feature order, label mapping, and headline metrics
- `metrics.json` reports holdout metrics and 5-fold stratified cross-validation

Reported metrics depend on the exact dataset and software versions. The training script prints the actual results; do not treat a result from one split as a guarantee of future performance.

## Local setup (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m uvicorn app:app --reload
```
Open http://127.0.0.1:8000/docs. Endpoints: `GET /`, `GET /health`, `GET /model-info`, `POST /predict`.

## Docker
```powershell
docker build -t breast-cancer-ml-api .
docker run --rm -p 8000:8000 breast-cancer-ml-api
```
Then visit http://localhost:8000/docs.

## GitHub
Create a new empty GitHub repository, then:
```bash
git init
git add .
git commit -m "Add breast cancer classification API"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/breast-cancer-ml-api.git
git push -u origin main
```
The GitHub Actions workflow installs dependencies, trains the model, runs pytest, and builds the Docker image. Confirm the run in the repository's Actions tab.

## Jupyter / Colab
Open `breast_cancer_training.ipynb`. In Colab, upload `data.csv` alongside the notebook or use the Files panel to add it, then run all cells.

## Important
This is a learning exercise, not a medical device or diagnostic service. Do not use predictions for diagnosis, treatment, or health decisions.
