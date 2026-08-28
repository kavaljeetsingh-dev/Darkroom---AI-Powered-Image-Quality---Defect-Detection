"""Tests for the /api/health endpoint."""
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_returns_200(client):
    response = client.get("/api/health")
    assert response.status_code == 200


def test_health_contains_required_fields(client):
    data = response = client.get("/api/health").json()
    assert "status" in data
    assert "model_loaded" in data
    assert "database_reachable" in data
    assert "version" in data


def test_health_model_loaded(client):
    data = client.get("/api/health").json()
    assert data["model_loaded"] is True


def test_health_database_reachable(client):
    data = client.get("/api/health").json()
    assert data["database_reachable"] is True
