from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.core.settings import AppSettings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.services.limix_adapter import (  # noqa: E402
    ClassificationOutput,
    RegressionOutput,
)


class DeterministicAdapter:
    def __init__(self) -> None:
        self.settings = type(
            "FakeSettings",
            (),
            {"model_path": Path("LimiX-2M.ckpt")},
        )()

    def classify(
        self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
    ) -> ClassificationOutput:
        classes = np.unique(y_train)
        scores = np.zeros((len(x_test), len(classes)), dtype=float)
        if len(classes) == 2:
            boundary = float(np.median(x_train[:, 0]))
            positive = (x_test[:, 0] >= boundary).astype(float)
            scores[:, 1] = 0.2 + 0.6 * positive
            scores[:, 0] = 1 - scores[:, 1]
        else:
            scores[:] = 1 / len(classes)
        predictions = classes[np.argmax(scores, axis=1)]
        return ClassificationOutput(predictions, scores, classes, 0.01, "cpu")

    def regress(
        self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
    ) -> RegressionOutput:
        prediction = np.full(len(x_test), float(np.mean(y_train)))
        return RegressionOutput(prediction, 0.01, "cpu")


@pytest.fixture
def app_settings(tmp_path: Path) -> AppSettings:
    artifacts = tmp_path / "artifacts"
    return AppSettings(
        root_dir=tmp_path,
        artifacts_dir=artifacts,
        database_path=artifacts / "workbench.db",
        max_upload_bytes=1024 * 1024,
        max_rows=1_000,
        max_source_features=50,
        max_model_features=100,
    )


@pytest.fixture
def adapter() -> DeterministicAdapter:
    return DeterministicAdapter()


@pytest.fixture
def client(app_settings: AppSettings, adapter: DeterministicAdapter) -> TestClient:
    return TestClient(create_app(app_settings, adapter), raise_server_exceptions=False)
