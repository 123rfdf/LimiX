from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Repository:
    def __init__(self, database_path: Path):
        self.database_path = database_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    inspection_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    dataset_id TEXT NOT NULL REFERENCES datasets(id),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id),
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    metrics_json TEXT,
                    result_path TEXT,
                    context_path TEXT,
                    error_message TEXT,
                    device TEXT,
                    inference_seconds REAL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );
                """
            )

    @staticmethod
    def _decode(row: sqlite3.Row | None, json_fields: tuple[str, ...]) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        for field in json_fields:
            if result.get(field):
                result[field.removesuffix("_json")] = json.loads(result.pop(field))
            else:
                result.pop(field, None)
                result[field.removesuffix("_json")] = None
        return result

    def add_dataset(
        self, dataset_id: str, filename: str, path: Path, sha256: str, inspection: dict[str, Any]
    ) -> dict[str, Any]:
        created_at = utc_now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?)",
                (dataset_id, filename, str(path), sha256, json.dumps(inspection), created_at),
            )
        return self.get_dataset(dataset_id)  # type: ignore[return-value]

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM datasets WHERE id = ?", (dataset_id,)
            ).fetchone()
        return self._decode(row, ("inspection_json",))

    def add_project(self, project_id: str, name: str, dataset_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO projects VALUES (?, ?, ?, ?)",
                (project_id, name, dataset_id, utc_now()),
            )
        return self.get_project(project_id)  # type: ignore[return-value]

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT p.*, d.filename AS dataset_filename, d.inspection_json
                FROM projects p JOIN datasets d ON d.id = p.dataset_id WHERE p.id = ?
                """,
                (project_id,),
            ).fetchone()
        return self._decode(row, ("inspection_json",))

    def list_projects(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.*, d.filename AS dataset_filename, d.inspection_json,
                       (SELECT COUNT(*) FROM runs r WHERE r.project_id = p.id) AS run_count
                FROM projects p JOIN datasets d ON d.id = p.dataset_id
                ORDER BY p.created_at DESC
                """
            ).fetchall()
        return [self._decode(row, ("inspection_json",)) for row in rows]  # type: ignore[misc]

    def add_run(self, run_id: str, project_id: str, config: dict[str, Any]) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (id, project_id, status, config_json, created_at)
                VALUES (?, ?, 'queued', ?, ?)
                """,
                (run_id, project_id, json.dumps(config), utc_now()),
            )
        return self.get_run(run_id)  # type: ignore[return-value]

    def update_run(self, run_id: str, **fields: Any) -> None:
        allowed = {
            "status",
            "metrics_json",
            "result_path",
            "context_path",
            "error_message",
            "device",
            "inference_seconds",
            "started_at",
            "completed_at",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unsupported run fields: {sorted(unknown)}")
        if not fields:
            return
        assignments = ", ".join(f"{key} = ?" for key in fields)
        values = [
            json.dumps(value) if key == "metrics_json" else value for key, value in fields.items()
        ]
        with self._connect() as connection:
            connection.execute(
                f"UPDATE runs SET {assignments} WHERE id = ?",  # noqa: S608 - fixed allowlist
                (*values, run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return self._decode(row, ("config_json", "metrics_json"))

    def list_runs(self, project_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM runs"
        params: tuple[str, ...] = ()
        if project_id:
            sql += " WHERE project_id = ?"
            params = (project_id,)
        sql += " ORDER BY created_at DESC"
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._decode(row, ("config_json", "metrics_json")) for row in rows]  # type: ignore[misc]
