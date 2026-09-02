from __future__ import annotations

import csv
import hashlib
import io
import shutil
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.errors import ValidationError
from app.core.settings import AppSettings
from app.models.repository import Repository


class DatasetService:
    def __init__(self, settings: AppSettings, repository: Repository):
        self.settings = settings
        self.repository = repository

    def inspect_and_store(self, filename: str, content: bytes) -> dict[str, Any]:
        if not filename.lower().endswith(".csv"):
            raise ValidationError("Only CSV files are supported.")
        if not content:
            raise ValidationError("The uploaded CSV file is empty.")
        if len(content) > self.settings.max_upload_bytes:
            max_megabytes = self.settings.max_upload_bytes // (1024 * 1024)
            raise ValidationError(
                f"CSV exceeds the {max_megabytes} MB upload limit."
            )
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValidationError("CSV must use UTF-8 or UTF-8 with BOM encoding.") from exc
        try:
            rows = csv.reader(io.StringIO(text))
            header = next(rows)
        except (StopIteration, csv.Error) as exc:
            raise ValidationError("CSV does not contain a valid header row.") from exc
        if not header or any(not name.strip() for name in header):
            raise ValidationError("Every CSV column must have a non-empty name.")
        duplicates = sorted({name for name in header if header.count(name) > 1})
        if duplicates:
            raise ValidationError("CSV contains duplicate column names: " + ", ".join(duplicates))
        try:
            frame = pd.read_csv(io.StringIO(text))
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
            raise ValidationError("CSV structure is invalid and could not be parsed.") from exc
        if frame.empty:
            raise ValidationError("CSV must contain at least one data row.")
        if frame.shape[1] < 2:
            raise ValidationError(
                "CSV must contain at least one feature column and one target column."
            )
        if len(frame) > self.settings.max_rows:
            row_count = len(frame)
            raise ValidationError(
                f"CSV has {row_count} rows; the configured model limit is {self.settings.max_rows}."
            )
        if frame.shape[1] - 1 > self.settings.max_source_features:
            raise ValidationError(
                f"CSV has {frame.shape[1]} columns; the configured feature limit is "
                f"{self.settings.max_source_features}."
            )
        numeric = frame.select_dtypes(include="number").columns.tolist()
        categorical = [name for name in frame.columns if name not in numeric]
        inspection = {
            "rows": len(frame),
            "columns": frame.shape[1],
            "column_names": frame.columns.tolist(),
            "numeric_columns": numeric,
            "categorical_columns": categorical,
            "missing_values": {key: int(value) for key, value in frame.isna().sum().items()},
            "duplicate_rows": int(frame.duplicated().sum()),
            "file_bytes": len(content),
            "encoding": "utf-8",
            "preview": frame.head(20).where(pd.notna(frame.head(20)), None).to_dict("records"),
        }
        dataset_id = str(uuid.uuid4())
        dataset_dir = self.settings.artifacts_dir / "datasets" / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=False)
        path = dataset_dir / "data.csv"
        path.write_bytes(content)
        record = self.repository.add_dataset(
            dataset_id,
            Path(filename).name,
            path,
            hashlib.sha256(content).hexdigest(),
            inspection,
        )
        return record

    def copy_example(self, source: Path) -> dict[str, Any]:
        return self.inspect_and_store(source.name, source.read_bytes())

    def remove_dataset_artifact(self, dataset_id: str) -> None:
        target = self.settings.artifacts_dir / "datasets" / dataset_id
        if target.exists() and target.parent == self.settings.artifacts_dir / "datasets":
            shutil.rmtree(target)
