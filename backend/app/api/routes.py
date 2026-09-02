from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.responses import FileResponse

from app.core.errors import ValidationError
from app.schemas.api import ProjectCreate, RunCreate

router = APIRouter(prefix="/api")


def services(request: Request) -> Any:
    return request.app.state.services


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    adapter = services(request).adapter
    return {
        "status": "ok",
        "application": "LimiX Workbench",
        "model_configured": adapter.settings.model_path.is_file(),
        "model": adapter.settings.model_path.name,
        "retrieval": False,
    }


@router.post("/datasets/inspect", status_code=201)
async def inspect_dataset(
    request: Request, file: Annotated[UploadFile, File()]
) -> dict[str, Any]:
    content = await file.read()
    return services(request).datasets.inspect_and_store(file.filename or "upload.csv", content)


@router.post("/projects", status_code=201)
def create_project(request: Request, payload: ProjectCreate) -> dict[str, Any]:
    service = services(request)
    if not service.repository.get_dataset(payload.dataset_id):
        raise ValidationError("Dataset does not exist.")
    return service.repository.add_project(str(uuid.uuid4()), payload.name, payload.dataset_id)


@router.get("/projects")
def list_projects(request: Request) -> list[dict[str, Any]]:
    return services(request).repository.list_projects()


@router.post("/runs", status_code=202)
def create_run(request: Request, payload: RunCreate) -> dict[str, Any]:
    return services(request).runs.create(payload)


@router.get("/runs")
def list_runs(
    request: Request, project_id: str | None = Query(default=None)
) -> list[dict[str, Any]]:
    return services(request).repository.list_runs(project_id)


@router.get("/runs/{run_id}")
def get_run(request: Request, run_id: str) -> dict[str, Any]:
    run = services(request).repository.get_run(run_id)
    if not run:
        raise ValidationError("Run does not exist.")
    return run


@router.get("/runs/{run_id}/results")
def get_results(request: Request, run_id: str) -> dict[str, Any]:
    run = services(request).repository.get_run(run_id)
    if not run:
        raise ValidationError("Run does not exist.")
    if run["status"] != "completed":
        raise ValidationError("Results are available only after the run completes.")
    return {
        "run_id": run_id,
        "metrics": run["metrics"],
        "device": run["device"],
        "inference_seconds": run["inference_seconds"],
    }


@router.post("/runs/{run_id}/predict")
async def predict(
    request: Request, run_id: str, file: Annotated[UploadFile, File()]
) -> FileResponse:
    content = await file.read()
    output = services(request).runs.predict_batch(run_id, file.filename or "batch.csv", content)
    return FileResponse(output, media_type="text/csv", filename=f"limix-batch-{run_id[:8]}.csv")


@router.get("/runs/{run_id}/download")
def download(request: Request, run_id: str) -> FileResponse:
    run = services(request).repository.get_run(run_id)
    if not run:
        raise ValidationError("Run does not exist.")
    path_text = run.get("result_path")
    if run["status"] != "completed" or not path_text or not Path(path_text).is_file():
        raise ValidationError("Prediction result file does not exist.")
    return FileResponse(
        path_text, media_type="text/csv", filename=f"limix-results-{run_id[:8]}.csv"
    )
