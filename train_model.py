
# from pathlib import Path
# import json
# import joblib
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.feature_selection import SequentialFeatureSelector
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# BASE_DIR = Path(__file__).resolve().parent
# DATA_PATH = BASE_DIR / "data.csv"
# MODEL_PATH = BASE_DIR / "model.joblib"
# METRICS_PATH = BASE_DIR / "metrics.json"

# def main():
#     df = pd.read_csv(DATA_PATH)
#     df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
#     df = df.dropna(subset=["diagnosis"])

#     y = df.pop("diagnosis").map({"B": 0, "M": 1})
#     if y.isna().any():
#         raise ValueError("Expected diagnosis labels B and M.")

#     feature_names = ["radius_mean", "texture_mean"]
#     X = df[feature_names].astype(float)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )

#     model = Pipeline([
#         ("scaler", StandardScaler()),
#         ("classifier", DecisionTreeClassifier(max_depth=2, random_state=42))
#     ])

#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
#     accuracy = accuracy_score(y_test, predictions)

#     bundle = {
#         "model": model,
#         "feature_names": feature_names,
#         "label_map": {0: "Benign", 1: "Malignant"}
#     }
#     joblib.dump(bundle, MODEL_PATH)

#     metrics = {
#         "model": "StandardScaler + DecisionTreeClassifier(max_depth=2)",
#         "features": feature_names,
#         "test_accuracy": float(accuracy),
#         "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
#         "classification_report": classification_report(
#             y_test, predictions,
#             target_names=["Benign", "Malignant"],
#             output_dict=True
#         )
#     }
#     METRICS_PATH.write_text(json.dumps(metrics, indent=2))

#     print(f"Test accuracy: {accuracy:.2%}")
#     print("Features:", feature_names)
#     print("Saved:", MODEL_PATH)

# if __name__ == "__main__":
#     main()

from pathlib import Path
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"

# MLflow configuration
mlflow.set_tracking_uri("http://127.0.0.1:6001")
mlflow.set_registry_uri("http://127.0.0.1:6001")
mlflow.set_experiment("Breast Cancer Classification")


def main():

    # 1. Load dataset
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    df = df.dropna(subset=["diagnosis"])

    y = df.pop("diagnosis").map({"B": 0, "M": 1})

    if y.isna().any():
        raise ValueError("Expected diagnosis labels B and M.")

    feature_names = ["radius_mean", "texture_mean"]
    X = df[feature_names].astype(float)

    # 2. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3. Create model
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", DecisionTreeClassifier(
            max_depth=2,
            random_state=42
        ))
    ])

    # 4. Start MLflow experiment run
    with mlflow.start_run(run_name="Decision Tree Model"):

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        predictions = model.predict(X_test)

        # 5. Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_test, predictions, average="weighted", zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, average="weighted", zero_division=0
        )

        cm = confusion_matrix(y_test, predictions)

        report = classification_report(
            y_test,
            predictions,
            target_names=["Benign", "Malignant"],
            output_dict=True
        )

        # 6. Log parameters to MLflow
        mlflow.log_params({
            "model_type": "DecisionTreeClassifier",
            "max_depth": 2,
            "random_state": 42,
            "test_size": 0.2,
            "scaler": "StandardScaler",
            "features": ", ".join(feature_names)
        })

        # 7. Log metrics to MLflow
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        # 8. Save existing model bundle
        bundle = {
            "model": model,
            "feature_names": feature_names,
            "label_map": {
                0: "Benign",
                1: "Malignant"
            }
        }

        joblib.dump(bundle, MODEL_PATH)

        # 9. Create metrics JSON
        metrics = {
            "model": "StandardScaler + DecisionTreeClassifier(max_depth=2)",
            "features": feature_names,
            "test_accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }

        METRICS_PATH.write_text(
            json.dumps(metrics, indent=2)
        )

        # 10. Log metrics file as MLflow artifact
        mlflow.log_artifact(str(METRICS_PATH))

        # Also log the confusion matrix as JSON
        mlflow.log_dict(
            {"confusion_matrix": cm.tolist()},
            "confusion_matrix.json"
        )

        # 11. Log and register trained model
        signature = infer_signature(
            X_train,
            model.predict(X_train)
        )

        mlflow.sklearn.log_model(
        sk_model=model,
        name="breast_cancer_model",
        signature=signature,
        input_example=X_train.head(3),
        registered_model_name="BreastCancerClassifier",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        print("\nMLflow run completed successfully!")
        print(f"Test accuracy: {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"F1-score: {f1:.2%}")
        print("Features:", feature_names)
        print("Saved model:", MODEL_PATH)
        print("Saved metrics:", METRICS_PATH)
        print("Registered model: BreastCancerClassifier")


if __name__ == "__main__":
    main()
