"""Basic health checks for the QueryNest API."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_root_endpoint_returns_ok():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_endpoint_returns_service_info():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "QueryNest"


def test_documents_endpoint_returns_structure():
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "sources" in data
