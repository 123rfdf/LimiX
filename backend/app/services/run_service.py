from __future__ import annotations

import csv
import io
import logging
import threading
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.core.errors import ValidationError, WorkbenchError
from app.core.settings import AppSettings
from app.models.repository import Repository, utc_now
from app.schemas.api import RunCreate
from app.services.limix_adapter import LimiXAdapter

LOGGER = logging.getLogger(__name__)


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


class RunService:
    def __init__(
        self,
        settings: AppSettings,
        repository: Repository,
        adapter: LimiXAdapter,
    ):
        self.settings = settings
        self.repository = repository
        self.adapter = adapter
        self._threads: dict[str, threading.Thread] = {}

    def create(self, request: RunCreate) -> dict[str, Any]:
        project = self.repository.get_project(request.project_id)
        if not project:
            raise ValidationError("The selected project does not exist.")
        run_id = str(uuid.uuid4())
        record = self.repository.add_run(run_id, request.project_id, request.model_dump())
        thread = threading.Thread(
            target=self._execute_safely,
            args=(run_id,),
            name=f"limix-run-{run_id[:8]}",
            daemon=True,
        )
        self._threads[run_id] = thread
        thread.start()
        return record

    def _execute_safely(self, run_id: str) -> None:
        self.repository.update_run(run_id, status="running", started_at=utc_now())
        try:
            self._execute(run_id)
        except WorkbenchError as exc:
            LOGGER.exception("Run %s failed", run_id)
            self.repository.update_run(
                run_id,
                status="failed",
                error_message=str(exc),
                completed_at=utc_now(),
            )
        except Exception:
            LOGGER.exception("Unexpected failure in run %s", run_id)
            self.repository.update_run(
                run_id,
                status="failed",
                error_message="The experiment failed unexpectedly. Check the backend log.",
                completed_at=utc_now(),
            )
        finally:
            self._threads.pop(run_id, None)

    def _execute(self, run_id: str) -> None:
        run = self.repository.get_run(run_id)
        if not run:
            raise ValidationError("Run record no longer exists.")
        project = self.repository.get_project(run["project_id"])
        if not project:
            raise ValidationError("Project no longer exists.")
        dataset = self.repository.get_dataset(project["dataset_id"])
        if not dataset or not Path(dataset["path"]).is_file():
            raise ValidationError("The project's dataset file is missing.")
        frame = pd.read_csv(dataset["path"])
        config = run["config"]
        target = config["target_column"]
        if target not in frame.columns:
            raise ValidationError(f"Target column '{target}' does not exist.")
        if frame[target].isna().any():
            raise ValidationError("Target column contains missing values.")
        feature_columns = config.get("feature_columns") or [
            column for column in frame.columns if column != target
        ]
        feature_columns = list(dict.fromkeys(feature_columns))
        if target in feature_columns:
            raise ValidationError("The target column cannot also be a feature.")
        missing_features = [column for column in feature_columns if column not in frame.columns]
        if missing_features:
            raise ValidationError("Feature columns do not exist: " + ", ".join(missing_features))
        if not feature_columns:
            raise ValidationError("Select at least one feature column.")
        if len(feature_columns) > self.settings.max_source_features:
            raise ValidationError(
                f"Selected feature count exceeds the limit of {self.settings.max_source_features}."
            )

        x = frame[feature_columns]
        y = frame[target]
        task = self._resolve_task(config["task_type"], y)
        self._validate_target(task, y)
        stratify = y if task == "classification" else None
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=config["test_size"],
                random_state=config["random_seed"],
                stratify=stratify,
            )
        except ValueError as exc:
            raise ValidationError(
                "The requested train/test split is too small for this target distribution."
            ) from exc
        if len(x_train) < 4 or len(x_test) < 2:
            raise ValidationError("Training or test split is too small for reliable inference.")

        preprocessor = self._build_preprocessor(x_train)
        try:
            x_train_model = np.asarray(preprocessor.fit_transform(x_train), dtype=np.float32)
            x_test_model = np.asarray(preprocessor.transform(x_test), dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValidationError("Selected feature types could not be preprocessed.") from exc
        if x_train_model.shape[1] > self.settings.max_model_features:
            raise ValidationError(
                f"Preprocessing produced {x_train_model.shape[1]} model features; the limit is "
                f"{self.settings.max_model_features}. Reduce high-cardinality categorical columns."
            )

        if task == "classification":
            metrics, predictions, context_extra, device, duration = self._classification(
                x_train_model, y_train.to_numpy(), x_test_model, y_test.to_numpy()
            )
        else:
            metrics, predictions, context_extra, device, duration = self._regression(
                x_train_model, y_train.to_numpy(), x_test_model, y_test.to_numpy()
            )
        metrics["split"] = {
            "training_rows": len(x_train),
            "test_rows": len(x_test),
            "source_features": len(feature_columns),
            "model_features": x_train_model.shape[1],
            "random_seed": config["random_seed"],
            "test_size": config["test_size"],
        }
        if task == "classification":
            metrics["target_distribution"] = {
                str(key): int(value) for key, value in y.value_counts().items()
            }

        run_dir = self.settings.artifacts_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        result_path = run_dir / "predictions.csv"
        context_path = run_dir / "context.joblib"
        predictions.to_csv(result_path, index=False)
        joblib.dump(
            {
                "task": task,
                "preprocessor": preprocessor,
                "feature_columns": feature_columns,
                "x_train": x_train_model,
                "y_train": y_train.to_numpy(),
                **context_extra,
            },
            context_path,
        )
        self.repository.update_run(
            run_id,
            status="completed",
            metrics_json=json_value(metrics),
            result_path=str(result_path),
            context_path=str(context_path),
            device=device,
            inference_seconds=duration,
            completed_at=utc_now(),
        )

    @staticmethod
    def _resolve_task(requested: str, target: pd.Series) -> str:
        if requested != "auto":
            return requested
        if pd.api.types.is_numeric_dtype(target):
            unique = target.nunique(dropna=True)
            if unique > max(20, int(len(target) * 0.05)):
                return "regression"
        return "classification"

    @staticmethod
    def _validate_target(task: str, target: pd.Series) -> None:
        unique = target.nunique(dropna=True)
        if task == "classification":
            if unique < 2:
                raise ValidationError("Classification target must contain at least two classes.")
            if unique > 20:
                raise ValidationError("Classification supports at most 20 target classes.")
            if int(target.value_counts().min()) < 2:
                raise ValidationError("Every class needs at least two rows for a stratified split.")
        else:
            try:
                numeric = pd.to_numeric(target)
            except (TypeError, ValueError) as exc:
                raise ValidationError("Regression target must be numeric.") from exc
            if not np.isfinite(numeric.to_numpy(dtype=float)).all():
                raise ValidationError("Regression target contains non-finite values.")
            if float(numeric.std()) == 0:
                raise ValidationError("Regression target must have non-zero variance.")

    @staticmethod
    def _build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
        numeric = frame.select_dtypes(include="number").columns.tolist()
        categorical = [column for column in frame.columns if column not in numeric]
        transformers: list[tuple[str, Any, list[str]]] = []
        if numeric:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric,
                )
            )
        if categorical:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ),
                        ]
                    ),
                    categorical,
                )
            )
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _classification(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any], str, float]:
        output = self.adapter.classify(x_train, y_train, x_test)
        baseline = LogisticRegression(max_iter=2_000, random_state=0)
        baseline.fit(x_train, y_train)
        baseline_prediction = baseline.predict(x_test)
        baseline_probability = baseline.predict_proba(x_test)
        metrics = {
            "task_type": "classification",
            "limix": self._classification_metrics(
                y_test, output.predictions, output.probabilities, output.classes
            ),
            "baseline": self._classification_metrics(
                y_test, baseline_prediction, baseline_probability, baseline.classes_
            ),
            "visualization": {
                "confusion_matrix": confusion_matrix(
                    y_test, output.predictions, labels=output.classes
                ).tolist(),
                "class_labels": [str(value) for value in output.classes],
            },
        }
        result = pd.DataFrame(
            {
                "actual": y_test,
                "limix_prediction": output.predictions,
                "baseline_prediction": baseline_prediction,
            }
        )
        for index, class_name in enumerate(output.classes):
            result[f"limix_probability_{class_name}"] = output.probabilities[:, index]
        return metrics, result, {}, output.device, output.inference_seconds

    @staticmethod
    def _classification_metrics(
        actual: np.ndarray,
        prediction: np.ndarray,
        probability: np.ndarray,
        classes: np.ndarray,
    ) -> dict[str, Any]:
        values: dict[str, Any] = {
            "accuracy": accuracy_score(actual, prediction),
            "precision_macro": precision_score(
                actual, prediction, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(actual, prediction, average="macro", zero_division=0),
            "f1_macro": f1_score(actual, prediction, average="macro", zero_division=0),
        }
        try:
            values["log_loss"] = log_loss(actual, probability, labels=classes)
            values["roc_auc"] = (
                roc_auc_score(actual, probability[:, 1])
                if len(classes) == 2
                else roc_auc_score(actual, probability, multi_class="ovr", labels=classes)
            )
        except ValueError:
            pass
        return values

    def _regression(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any], str, float]:
        output = self.adapter.regress(x_train, y_train, x_test)
        baseline = Ridge(alpha=1.0)
        baseline.fit(x_train, y_train)
        baseline_prediction = baseline.predict(x_test)
        metrics = {
            "task_type": "regression",
            "limix": self._regression_metrics(y_test, output.predictions),
            "baseline": self._regression_metrics(y_test, baseline_prediction),
            "visualization": {
                "actual": y_test.tolist(),
                "predicted": output.predictions.tolist(),
                "residuals": (y_test - output.predictions).tolist(),
            },
        }
        result = pd.DataFrame(
            {
                "actual": y_test,
                "limix_prediction": output.predictions,
                "baseline_prediction": baseline_prediction,
                "limix_residual": y_test - output.predictions,
            }
        )
        return metrics, result, {}, output.device, output.inference_seconds

    @staticmethod
    def _regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
        return {
            "rmse": float(mean_squared_error(actual, prediction) ** 0.5),
            "mae": float(mean_absolute_error(actual, prediction)),
            "r2": float(r2_score(actual, prediction)),
        }

    def predict_batch(self, run_id: str, filename: str, content: bytes) -> Path:
        run = self.repository.get_run(run_id)
        if not run:
            raise ValidationError("Run does not exist.")
        if run["status"] != "completed" or not run.get("context_path"):
            raise ValidationError("Batch prediction requires a completed run.")
        context_path = Path(run["context_path"])
        if not context_path.is_file():
            raise ValidationError("The saved run context is missing.")
        frame = self._read_batch_csv(filename, content)
        context = joblib.load(context_path)
        expected = context["feature_columns"]
        missing = [column for column in expected if column not in frame.columns]
        extra = [column for column in frame.columns if column not in expected]
        if missing or extra:
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if extra:
                details.append("unexpected: " + ", ".join(extra))
            raise ValidationError(
                "Batch columns do not match the experiment (" + "; ".join(details) + ")."
            )
        try:
            transformed = np.asarray(
                context["preprocessor"].transform(frame[expected]), dtype=np.float32
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError("Batch features could not be transformed.") from exc
        result = frame.copy()
        if context["task"] == "classification":
            output = self.adapter.classify(context["x_train"], context["y_train"], transformed)
            result["prediction"] = output.predictions
            for index, class_name in enumerate(output.classes):
                result[f"probability_{class_name}"] = output.probabilities[:, index]
        else:
            output = self.adapter.regress(context["x_train"], context["y_train"], transformed)
            result["prediction"] = output.predictions
        run_dir = context_path.parent
        output_path = run_dir / f"batch-{uuid.uuid4().hex[:10]}.csv"
        result.to_csv(output_path, index=False)
        return output_path

    def _read_batch_csv(self, filename: str, content: bytes) -> pd.DataFrame:
        if not filename.lower().endswith(".csv"):
            raise ValidationError("Only CSV files are supported for batch prediction.")
        if not content:
            raise ValidationError("The batch CSV is empty.")
        if len(content) > self.settings.max_upload_bytes:
            raise ValidationError("The batch CSV exceeds the upload limit.")
        try:
            text = content.decode("utf-8-sig")
            header = next(csv.reader(io.StringIO(text)))
            if len(header) != len(set(header)):
                raise ValidationError("Batch CSV contains duplicate column names.")
            frame = pd.read_csv(io.StringIO(text))
        except UnicodeDecodeError as exc:
            raise ValidationError("Batch CSV must use UTF-8 encoding.") from exc
        except (StopIteration, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValidationError("Batch CSV is invalid.") from exc
        if frame.empty:
            raise ValidationError("Batch CSV must contain at least one row.")
        if len(frame) > self.settings.max_rows:
            raise ValidationError("Batch CSV exceeds the row limit.")
        return frame
