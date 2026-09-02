export interface Inspection {
  rows: number;
  columns: number;
  column_names: string[];
  numeric_columns: string[];
  categorical_columns: string[];
  missing_values: Record<string, number>;
  duplicate_rows: number;
  file_bytes: number;
  encoding: string;
  preview: Record<string, unknown>[];
}

export interface Dataset {
  id: string;
  filename: string;
  inspection: Inspection;
  created_at: string;
}

export interface Project {
  id: string;
  name: string;
  dataset_id: string;
  dataset_filename: string;
  inspection: Inspection;
  run_count?: number;
  created_at: string;
}

export type TaskType = "auto" | "classification" | "regression";

export interface RunConfig {
  project_id: string;
  target_column: string;
  feature_columns: string[];
  task_type: TaskType;
  test_size: number;
  random_seed: number;
}

export interface Run {
  id: string;
  project_id: string;
  status: "queued" | "running" | "completed" | "failed";
  config: RunConfig;
  metrics: Metrics | null;
  error_message: string | null;
  device: string | null;
  inference_seconds: number | null;
  created_at: string;
}

export interface Metrics {
  task_type: "classification" | "regression";
  limix: Record<string, number>;
  baseline: Record<string, number>;
  visualization: {
    confusion_matrix?: number[][];
    class_labels?: string[];
    actual?: number[];
    predicted?: number[];
    residuals?: number[];
  };
  split: Record<string, number>;
  target_distribution?: Record<string, number>;
}

class ApiError extends Error {
  code: string;

  constructor(code: string, message: string) {
    super(message);
    this.code = code;
  }
}

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    let code = "request_failed";
    try {
      const payload = (await response.json()) as {
        error?: { code?: string; message?: string };
      };
      message = payload.error?.message ?? message;
      code = payload.error?.code ?? code;
    } catch {
      // Keep the HTTP fallback message for non-JSON failures.
    }
    throw new ApiError(code, message);
  }
  return (await response.json()) as T;
}

export async function inspectDataset(file: File): Promise<Dataset> {
  const body = new FormData();
  body.append("file", file);
  return request<Dataset>("/api/datasets/inspect", { method: "POST", body });
}

export async function createProject(name: string, datasetId: string): Promise<Project> {
  return request<Project>("/api/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, dataset_id: datasetId }),
  });
}

export const listProjects = (): Promise<Project[]> => request<Project[]>("/api/projects");

export async function createRun(config: RunConfig): Promise<Run> {
  return request<Run>("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
}

export const getRun = (runId: string): Promise<Run> => request<Run>(`/api/runs/${runId}`);

export const listRuns = (projectId?: string): Promise<Run[]> =>
  request<Run[]>(`/api/runs${projectId ? `?project_id=${encodeURIComponent(projectId)}` : ""}`);

export async function predictBatch(runId: string, file: File): Promise<Blob> {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch(`/api/runs/${runId}/predict`, { method: "POST", body });
  if (!response.ok) {
    const payload = (await response.json()) as { error?: { message?: string } };
    throw new ApiError("batch_prediction_failed", payload.error?.message ?? "Batch prediction failed.");
  }
  return response.blob();
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

