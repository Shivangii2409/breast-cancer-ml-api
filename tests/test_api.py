from fastapi.testclient import TestClient
from app import app, FEATURE_NAMES

client = TestClient(app)

def sample_features():
    # Valid example values; these are illustrative inputs, not medical advice.
    values = {
        "radius_mean": 14.0, "texture_mean": 20.0, "perimeter_mean": 90.0,
        "area_mean": 600.0, "smoothness_mean": 0.1, "compactness_mean": 0.1,
        "concavity_mean": 0.08, "concave points_mean": 0.04, "symmetry_mean": 0.18,
        "fractal_dimension_mean": 0.06, "radius_se": 0.3, "texture_se": 1.0,
        "perimeter_se": 2.0, "area_se": 25.0, "smoothness_se": 0.006,
        "compactness_se": 0.02, "concavity_se": 0.02, "concave points_se": 0.01,
        "symmetry_se": 0.02, "fractal_dimension_se": 0.003, "radius_worst": 16.0,
        "texture_worst": 25.0, "perimeter_worst": 105.0, "area_worst": 800.0,
        "smoothness_worst": 0.13, "compactness_worst": 0.25,
        "concavity_worst": 0.25, "concave points_worst": 0.1,
        "symmetry_worst": 0.3, "fractal_dimension_worst": 0.08
    }
    return values

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["feature_count"] == 30

def test_prediction():
    response = client.post("/predict", json={"features": sample_features()})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in ["Benign", "Malignant"]
    assert 0 <= body["confidence"] <= 1

def test_missing_feature_rejected():
    values = sample_features()
    values.pop(FEATURE_NAMES[0])
    response = client.post("/predict", json={"features": values})
    assert response.status_code == 422
