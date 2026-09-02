from __future__ import annotations


class WorkbenchError(Exception):
    """Expected application error safe to translate at the API boundary."""

    code = "workbench_error"


class ConfigurationError(WorkbenchError):
    code = "configuration_error"


class ValidationError(WorkbenchError):
    code = "validation_error"


class InferenceError(WorkbenchError):
    code = "inference_error"


class ModelOutOfMemoryError(InferenceError):
    code = "model_out_of_memory"
