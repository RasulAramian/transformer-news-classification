from fastapi.testclient import TestClient

from src.api import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
    }


def test_predict_endpoint():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "text": (
                    "The stock market rose after the government "
                    "announced a new economic policy."
                ),
                "model": "distilbert",
            },
        )

    assert response.status_code == 200

    result = response.json()

    assert result["model"] == "distilbert"
    assert result["label"] in {
        "World",
        "Sports",
        "Business",
        "Sci/Tech",
    }
    assert 0.0 <= result["confidence"] <= 1.0
