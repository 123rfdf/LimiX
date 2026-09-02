from __future__ import annotations

import io
import time

import pandas as pd
from app.core.settings import AppSettings
from app.main import create_app
from conftest import DeterministicAdapter
from fastapi.testclient import TestClient


def upload(client: TestClient, frame: pd.DataFrame, filename: str = "data.csv") -> dict:
    response = client.post(
        "/api/datasets/inspect",
        files={"file": (filename, frame.to_csv(index=False).encode(), "text/csv")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def create_project(client: TestClient, dataset_id: str, name: str = "Demo") -> dict:
    response = client.post("/api/projects", json={"name": name, "dataset_id": dataset_id})
    assert response.status_code == 201, response.text
    return response.json()


def wait_for_run(client: TestClient, run_id: str) -> dict:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        run = response.json()
        if run["status"] in {"completed", "failed"}:
            return run
        time.sleep(0.03)
    raise AssertionError("Run did not finish")


def test_health_and_invalid_csv(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["retrieval"] is False

    invalid = client.post(
        "/api/datasets/inspect",
        files={"file": ("bad.csv", b"a,a\n1,2\n", "text/csv")},
    )
    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "validation_error"
    assert "duplicate column" in invalid.json()["error"]["message"].lower()


def test_classification_run_batch_and_download(client: TestClient) -> None:
    frame = pd.DataFrame(
        {
            "amount": list(range(20)),
            "segment": ["north", "south"] * 10,
            "target": ["no", "yes"] * 10,
        }
    )
    dataset = upload(client, frame)
    assert dataset["inspection"]["rows"] == 20
    assert dataset["inspection"]["categorical_columns"] == ["segment", "target"]
    project = create_project(client, dataset["id"])
    created = client.post(
        "/api/runs",
        json={
            "project_id": project["id"],
            "target_column": "target",
            "feature_columns": ["amount", "segment"],
            "task_type": "classification",
            "test_size": 0.3,
            "random_seed": 9,
        },
    )
    assert created.status_code == 202
    run = wait_for_run(client, created.json()["id"])
    assert run["status"] == "completed", run

    results = client.get(f"/api/runs/{run['id']}/results")
    assert results.status_code == 200
    assert results.json()["metrics"]["task_type"] == "classification"
    assert "accuracy" in results.json()["metrics"]["limix"]

    download = client.get(f"/api/runs/{run['id']}/download")
    assert download.status_code == 200
    assert "limix_prediction" in download.text

    batch = pd.DataFrame({"amount": [21, 22], "segment": ["north", "new"]})
    predicted = client.post(
        f"/api/runs/{run['id']}/predict",
        files={"file": ("batch.csv", batch.to_csv(index=False).encode(), "text/csv")},
    )
    assert predicted.status_code == 200, predicted.text
    output = pd.read_csv(io.BytesIO(predicted.content))
    assert list(output["amount"]) == [21, 22]
    assert "prediction" in output
    assert any(column.startswith("probability_") for column in output.columns)


def test_regression_and_sqlite_history_survives_restart(
    app_settings: AppSettings, adapter: DeterministicAdapter
) -> None:
    first = TestClient(create_app(app_settings, adapter), raise_server_exceptions=False)
    frame = pd.DataFrame(
        {
            "x": list(range(30)),
            "category": ["a", "b", "c"] * 10,
            "target": [value * 1.5 + 2 for value in range(30)],
        }
    )
    dataset = upload(first, frame, "regression.csv")
    project = create_project(first, dataset["id"], "Persistent project")
    response = first.post(
        "/api/runs",
        json={
            "project_id": project["id"],
            "target_column": "target",
            "task_type": "regression",
            "test_size": 0.2,
            "random_seed": 3,
        },
    )
    run = wait_for_run(first, response.json()["id"])
    assert run["status"] == "completed", run
    assert "rmse" in run["metrics"]["limix"]

    restarted = TestClient(create_app(app_settings, adapter), raise_server_exceptions=False)
    projects = restarted.get("/api/projects").json()
    runs = restarted.get("/api/runs", params={"project_id": project["id"]}).json()
    assert [item["name"] for item in projects] == ["Persistent project"]
    assert runs[0]["id"] == run["id"]
    assert runs[0]["status"] == "completed"


def test_batch_column_mismatch_is_clear(client: TestClient) -> None:
    frame = pd.DataFrame({"x": range(12), "target": [0, 1] * 6})
    dataset = upload(client, frame)
    project = create_project(client, dataset["id"])
    created = client.post(
        "/api/runs",
        json={
            "project_id": project["id"],
            "target_column": "target",
            "task_type": "classification",
            "test_size": 0.25,
        },
    )
    run = wait_for_run(client, created.json()["id"])
    assert run["status"] == "completed"
    mismatch = client.post(
        f"/api/runs/{run['id']}/predict",
        files={"file": ("batch.csv", b"wrong\n1\n", "text/csv")},
    )
    assert mismatch.status_code == 400
    assert "do not match" in mismatch.json()["error"]["message"]
