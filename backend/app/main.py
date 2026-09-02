from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.errors import WorkbenchError
from app.core.settings import AppSettings
from app.models.repository import Repository
from app.services.dataset_service import DatasetService
from app.services.limix_adapter import AdapterSettings, LimiXAdapter
from app.services.run_service import RunService

LOGGER = logging.getLogger(__name__)


@dataclass
class Services:
    settings: AppSettings
    repository: Repository
    adapter: LimiXAdapter
    datasets: DatasetService
    runs: RunService


def build_services(settings: AppSettings | None = None, adapter: Any | None = None) -> Services:
    settings = settings or AppSettings.from_environment()
    settings.prepare()
    repository = Repository(settings.database_path)
    repository.initialize()
    adapter = adapter or LimiXAdapter(AdapterSettings.from_environment())
    return Services(
        settings=settings,
        repository=repository,
        adapter=adapter,
        datasets=DatasetService(settings, repository),
        runs=RunService(settings, repository, adapter),
    )


def create_app(settings: AppSettings | None = None, adapter: Any | None = None) -> FastAPI:
    app = FastAPI(title="LimiX Workbench API", version="1.0.0")
    app.state.services = build_services(settings, adapter)

    @app.exception_handler(WorkbenchError)
    async def workbench_error_handler(_request: Request, exc: WorkbenchError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": {"code": exc.code, "message": str(exc), "details": None}},
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "request_validation_error",
                    "message": "The request is invalid.",
                    "details": exc.errors(),
                }
            },
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("Unhandled API error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "The server encountered an unexpected error. Check the backend log.",
                    "details": None,
                }
            },
        )

    app.include_router(router)
    frontend = app.state.services.settings.root_dir / "frontend" / "dist"
    if frontend.is_dir():
        assets = frontend / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            requested = (frontend / path).resolve()
            if requested.is_file() and frontend in requested.parents:
                return FileResponse(requested)
            return FileResponse(frontend / "index.html")

    return app


app = create_app()


if os.getenv("WORKBENCH_LOG_LEVEL"):
    logging.basicConfig(level=os.getenv("WORKBENCH_LOG_LEVEL", "INFO"))
