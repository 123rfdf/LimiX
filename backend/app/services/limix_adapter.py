from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.preprocessing import LabelEncoder

from app.core.errors import (
    ConfigurationError,
    InferenceError,
    ModelOutOfMemoryError,
    ValidationError,
)

TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class AdapterSettings:
    source_dir: Path
    model_path: Path
    classification_config: Path
    regression_config: Path
    preferred_device: Literal["auto", "cuda", "cpu"] = "auto"
    allow_cpu_fallback: bool = True

    @classmethod
    def from_environment(cls) -> AdapterSettings:
        source = Path(os.getenv("LIMIX_SOURCE_DIR", "__missing_limix_source__")).expanduser()
        model = Path(
            os.getenv("LIMIX_MODEL_PATH", str(source / "cache" / "LimiX-2M.ckpt"))
        ).expanduser()
        cls_config = Path(
            os.getenv(
                "LIMIX_CLASSIFICATION_CONFIG",
                str(source / "config" / "cls_default_noretrieval.json"),
            )
        ).expanduser()
        reg_config = Path(
            os.getenv(
                "LIMIX_REGRESSION_CONFIG",
                str(source / "config" / "reg_default_noretrieval.json"),
            )
        ).expanduser()
        device = os.getenv("LIMIX_DEVICE", "auto").lower()
        if device not in {"auto", "cuda", "cpu"}:
            raise ConfigurationError("LIMIX_DEVICE must be auto, cuda, or cpu.")
        return cls(
            source_dir=source,
            model_path=model,
            classification_config=cls_config,
            regression_config=reg_config,
            preferred_device=device,  # type: ignore[arg-type]
            allow_cpu_fallback=os.getenv("LIMIX_CPU_FALLBACK", "true").lower()
            in {"1", "true", "yes"},
        )


@dataclass(frozen=True)
class ClassificationOutput:
    predictions: np.ndarray
    probabilities: np.ndarray
    classes: np.ndarray
    inference_seconds: float
    device: str


@dataclass(frozen=True)
class RegressionOutput:
    predictions: np.ndarray
    inference_seconds: float
    device: str


class LimiXAdapter:
    """Small, explicit boundary around the upstream LimiX inference API."""

    def __init__(self, settings: AdapterSettings):
        self.settings = settings
        self._predictors: dict[tuple[TaskType, str], Any] = {}
        self._lock = threading.RLock()
        self._torch: Any | None = None
        self._predictor_type: Any | None = None

    def _validate_settings(self) -> None:
        required = {
            "LimiX source directory": self.settings.source_dir,
            "LimiX-2M checkpoint": self.settings.model_path,
            "classification no-retrieval config": self.settings.classification_config,
            "regression no-retrieval config": self.settings.regression_config,
        }
        missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
        if missing:
            raise ConfigurationError("Missing required local LimiX files: " + "; ".join(missing))
        if "2m" not in self.settings.model_path.name.lower():
            raise ConfigurationError("LimiX Workbench only supports a LimiX-2M checkpoint.")

    def _load_upstream(self) -> None:
        if self._predictor_type is not None:
            return
        self._validate_settings()
        source_text = str(self.settings.source_dir.resolve())
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        try:
            import torch
            from inference.predictor import LimiXPredictor
        except Exception as exc:  # pragma: no cover - depends on local upstream install
            raise ConfigurationError(
                "Unable to import the upstream LimiX predictor. Check LIMIX_SOURCE_DIR "
                "and install its dependencies in the active Python environment."
            ) from exc
        self._torch = torch
        self._predictor_type = LimiXPredictor

    def _select_device(self) -> str:
        self._load_upstream()
        assert self._torch is not None
        cuda_available = bool(self._torch.cuda.is_available())
        requested = self.settings.preferred_device
        if requested == "cpu":
            return "cpu"
        if cuda_available:
            return "cuda"
        if requested == "cuda" and not self.settings.allow_cpu_fallback:
            raise ConfigurationError("CUDA was requested but is not available.")
        return "cpu"

    def _get_predictor(self, task: TaskType, device: str) -> Any:
        self._load_upstream()
        key = (task, device)
        with self._lock:
            if key not in self._predictors:
                assert self._torch is not None and self._predictor_type is not None
                config = (
                    self.settings.classification_config
                    if task == "classification"
                    else self.settings.regression_config
                )
                self._predictors[key] = self._predictor_type(
                    device=self._torch.device(device),
                    model_path=str(self.settings.model_path),
                    inference_config=str(config),
                    inference_with_DDP=False,
                )
            return self._predictors[key]

    @staticmethod
    def _as_numeric_matrix(values: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValidationError(f"{name} must be a non-empty two-dimensional matrix.")
        try:
            numeric = array.astype(np.float32, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{name} contains unsupported non-numeric values.") from exc
        if not np.isfinite(numeric).all():
            raise ValidationError(f"{name} contains missing or non-finite values.")
        return numeric

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    @staticmethod
    def _is_oom(exc: BaseException) -> bool:
        text = str(exc).lower()
        return "out of memory" in text or "cuda error: out of memory" in text

    def _predict_with_fallback(
        self,
        task: TaskType,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
    ) -> tuple[np.ndarray, float, str]:
        device = self._select_device()
        started = time.perf_counter()
        try:
            predictor = self._get_predictor(task, device)
            result = predictor.predict(
                x_train,
                y_train,
                x_test,
                task_type="Classification" if task == "classification" else "Regression",
            )
            return self._to_numpy(result), time.perf_counter() - started, device
        except RuntimeError as exc:
            if device == "cuda" and self._is_oom(exc):
                assert self._torch is not None
                self._torch.cuda.empty_cache()
                if not self.settings.allow_cpu_fallback:
                    raise ModelOutOfMemoryError(
                        "CUDA ran out of memory and CPU fallback is disabled."
                    ) from exc
                cpu_started = time.perf_counter()
                try:
                    result = self._get_predictor(task, "cpu").predict(
                        x_train,
                        y_train,
                        x_test,
                        task_type=("Classification" if task == "classification" else "Regression"),
                    )
                    return self._to_numpy(result), time.perf_counter() - cpu_started, "cpu"
                except Exception as cpu_exc:
                    raise InferenceError(
                        "LimiX inference failed after CUDA memory exhaustion and CPU fallback."
                    ) from cpu_exc
            raise InferenceError("LimiX inference failed.") from exc
        except Exception as exc:
            raise InferenceError("LimiX inference failed.") from exc

    def classify(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
    ) -> ClassificationOutput:
        train = self._as_numeric_matrix(x_train, "x_train")
        test = self._as_numeric_matrix(x_test, "x_test")
        if train.shape[1] != test.shape[1]:
            raise ValidationError("Training and prediction feature counts do not match.")
        targets = np.asarray(y_train)
        if targets.ndim != 1 or len(targets) != len(train):
            raise ValidationError("y_train must contain one label per training row.")
        encoder = LabelEncoder().fit(targets)
        if len(encoder.classes_) < 2:
            raise ValidationError("Classification requires at least two target classes.")
        encoded = encoder.transform(targets).astype(np.int64, copy=False)
        probabilities, duration, device = self._predict_with_fallback(
            "classification", train, encoded, test
        )
        if probabilities.ndim != 2 or probabilities.shape != (
            len(test),
            len(encoder.classes_),
        ):
            raise InferenceError("LimiX returned an invalid classification probability matrix.")
        labels = encoder.inverse_transform(np.argmax(probabilities, axis=1))
        return ClassificationOutput(
            predictions=labels,
            probabilities=probabilities,
            classes=encoder.classes_,
            inference_seconds=duration,
            device=device,
        )

    def regress(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
    ) -> RegressionOutput:
        train = self._as_numeric_matrix(x_train, "x_train")
        test = self._as_numeric_matrix(x_test, "x_test")
        if train.shape[1] != test.shape[1]:
            raise ValidationError("Training and prediction feature counts do not match.")
        try:
            targets = np.asarray(y_train, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValidationError("Regression targets must be numeric.") from exc
        if targets.ndim != 1 or len(targets) != len(train) or not np.isfinite(targets).all():
            raise ValidationError("y_train must contain one finite numeric value per row.")
        mean = float(targets.mean())
        std = float(targets.std())
        if std <= np.finfo(np.float32).eps:
            raise ValidationError("Regression targets must have non-zero variance.")
        normalized = ((targets - mean) / std).astype(np.float32, copy=False)
        raw, duration, device = self._predict_with_fallback("regression", train, normalized, test)
        # Upstream CUDA inference may return float16. Cast before inverse scaling;
        # otherwise ordinary currency-like targets can overflow at ~65,504.
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)
        if len(raw) != len(test):
            raise InferenceError("LimiX returned an invalid regression prediction vector.")
        predictions = raw * std + mean
        if not np.isfinite(predictions).all():
            raise InferenceError("LimiX returned non-finite regression predictions.")
        return RegressionOutput(
            predictions=predictions,
            inference_seconds=duration,
            device=device,
        )
