"""End-to-end API tests using sample images."""
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "sample_images"


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import get_db, Base

@pytest.fixture
def client(tmp_path):
    db_file = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _analyze(client, filename: str) -> dict:
    path = SAMPLE_DIR / filename
    if not path.exists():
        pytest.skip(f"Sample image not found: {path}")
    with open(path, "rb") as f:
        response = client.post("/api/analyze", files={"file": (filename, f, "image/jpeg")})
    assert response.status_code == 200
    return response.json()


# ------------------------------------------------------------------
# POST /api/analyze
# ------------------------------------------------------------------

def test_analyze_good_quality(client):
    data = _analyze(client, "good_quality.jpg")
    assert data["quality_label"] == "ACCEPTABLE"
    assert data["recommended_action"] == "PASS"
    assert data["quality_score"] >= 78
    assert len(data["issues"]) == 0


def test_analyze_blurry(client):
    data = _analyze(client, "blurry.jpg")
    assert data["quality_label"] in ("DEGRADED", "DEFECTIVE")
    issue_types = [i["type"] for i in data["issues"]]
    assert "blur" in issue_types


def test_analyze_noisy(client):
    data = _analyze(client, "noisy.jpg")
    issue_types = [i["type"] for i in data["issues"]]
    assert "noise" in issue_types


def test_analyze_underexposed(client):
    data = _analyze(client, "underexposed_dark.jpg")
    issue_types = [i["type"] for i in data["issues"]]
    assert "underexposure" in issue_types


def test_analyze_overexposed(client):
    data = _analyze(client, "overexposed_bright.jpg")
    issue_types = [i["type"] for i in data["issues"]]
    assert "overexposure" in issue_types


def test_analyze_corrupted(client):
    data = _analyze(client, "corrupted_defective.jpg")
    issue_types = [i["type"] for i in data["issues"]]
    assert "corruption" in issue_types
    assert data["quality_label"] == "DEFECTIVE"


def test_analyze_has_recommended_action(client):
    data = _analyze(client, "good_quality.jpg")
    assert data["recommended_action"] in ("PASS", "REVIEW", "REJECT")


def test_analyze_has_image_path(client):
    data = _analyze(client, "good_quality.jpg")
    assert data["image_path"] is not None
    assert data["image_path"].startswith("/static/uploads/")


def test_analyze_issues_have_evidence(client):
    data = _analyze(client, "blurry.jpg")
    for issue in data["issues"]:
        if issue["type"] != "potential_defect":
            assert len(issue["evidence"]) > 0, f"Issue {issue['type']} has no evidence"


# ------------------------------------------------------------------
# GET /api/results  &  GET /api/results/:id
# ------------------------------------------------------------------

def test_list_results(client):
    # Ensure at least one result exists
    _analyze(client, "good_quality.jpg")
    response = client.get("/api/results?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "results" in data
    assert len(data["results"]) > 0


def test_get_result_by_id(client):
    created = _analyze(client, "good_quality.jpg")
    result_id = created["id"]
    response = client.get(f"/api/results/{result_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == result_id
    assert data["image_path"] is not None


def test_get_nonexistent_result(client):
    response = client.get("/api/results/999999")
    assert response.status_code == 404


# ------------------------------------------------------------------
# DELETE /api/results/:id
# ------------------------------------------------------------------

def test_delete_result(client):
    created = _analyze(client, "good_quality.jpg")
    result_id = created["id"]
    response = client.delete(f"/api/results/{result_id}")
    assert response.status_code == 200
    assert response.json()["deleted"] == result_id

    # Verify it's gone
    response = client.get(f"/api/results/{result_id}")
    assert response.status_code == 404
