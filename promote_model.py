
import mlflow
from mlflow import MlflowClient

MODEL_NAME = "BreastCancerClassifier"

ACCURACY_THRESHOLD = 0.85
F1_THRESHOLD = 0.80

mlflow.set_tracking_uri("http://127.0.0.1:6001")
mlflow.set_registry_uri("http://127.0.0.1:6001")

client = MlflowClient()


def promote_model():

    # Get all registered versions
    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    if not versions:
        raise ValueError("No registered model versions found.")

    # Find latest version
    latest_version = max(
        versions,
        key=lambda version: int(version.version)
    )

    version_number = latest_version.version

    # Retrieve metrics from the training run
    run = client.get_run(latest_version.run_id)
    metrics = run.data.metrics

    accuracy = metrics.get("accuracy", 0)
    f1_score = metrics.get("f1_score", 0)

    print(f"\nEvaluating {MODEL_NAME} v{version_number}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"F1-score: {f1_score:.2%}")

    # Validation gate
    if (
        accuracy >= ACCURACY_THRESHOLD
        and f1_score >= F1_THRESHOLD
    ):

        # Promote using champion alias
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="champion",
            version=version_number
        )

        print("\nVALIDATION PASSED")
        print(f"{MODEL_NAME} v{version_number} is now CHAMPION!")

    else:
        print("\nVALIDATION FAILED")
        print("Model was NOT promoted.")
        print("Existing champion remains unchanged.")


if __name__ == "__main__":
    promote_model()
