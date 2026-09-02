from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    dataset_id: str = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Project name cannot be blank.")
        return value


class RunCreate(BaseModel):
    project_id: str
    target_column: str
    feature_columns: list[str] | None = None
    task_type: Literal["auto", "classification", "regression"] = "auto"
    test_size: float = Field(default=0.2, ge=0.1, le=0.8)
    random_seed: int = Field(default=42, ge=0, le=2_147_483_647)


class ErrorBody(BaseModel):
    code: str
    message: str
    details: object | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
