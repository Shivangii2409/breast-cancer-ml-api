
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"

def main():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    df = df.dropna(subset=["diagnosis"])

    y = df.pop("diagnosis").map({"B": 0, "M": 1})
    if y.isna().any():
        raise ValueError("Expected diagnosis labels B and M.")

    feature_names = ["radius_mean", "texture_mean"]
    X = df[feature_names].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", DecisionTreeClassifier(max_depth=2, random_state=42))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    bundle = {
        "model": model,
        "feature_names": feature_names,
        "label_map": {0: "Benign", 1: "Malignant"}
    }
    joblib.dump(bundle, MODEL_PATH)

    metrics = {
        "model": "StandardScaler + DecisionTreeClassifier(max_depth=2)",
        "features": feature_names,
        "test_accuracy": float(accuracy),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions,
            target_names=["Benign", "Malignant"],
            output_dict=True
        )
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"Test accuracy: {accuracy:.2%}")
    print("Features:", feature_names)
    print("Saved:", MODEL_PATH)

if __name__ == "__main__":
    main()
