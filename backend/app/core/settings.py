from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    root_dir: Path
    artifacts_dir: Path
    database_path: Path
    max_upload_bytes: int = 25 * 1024 * 1024
    max_rows: int = 10_000
    max_source_features: int = 100
    max_model_features: int = 500

    @classmethod
    def from_environment(cls) -> AppSettings:
        root = Path(__file__).resolve().parents[3]
        artifacts = Path(os.getenv("WORKBENCH_ARTIFACTS_DIR", root / "artifacts")).resolve()
        database = Path(os.getenv("WORKBENCH_DATABASE_PATH", artifacts / "workbench.db")).resolve()
        return cls(root_dir=root, artifacts_dir=artifacts, database_path=database)

    def prepare(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
