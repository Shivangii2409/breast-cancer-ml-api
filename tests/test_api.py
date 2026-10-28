
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


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
    assert response.json()["features"] == [
        "radius_mean",
        "texture_mean"
    ]


def test_prediction():
    response = client.post(
        "/predict",
        json={
            "radius_mean": 14.0,
            "texture_mean": 20.0
        }
    )
    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in [0, 1]
    assert body["diagnosis"] in ["Benign", "Malignant"]
    assert 0 <= body["confidence"] <= 1


def test_invalid_input():
    response = client.post(
        "/predict",
        json={
            "radius_mean": -1,
            "texture_mean": 20.0
        }
    )
    assert response.status_code == 422
