# API

All errors use `{"error":{"code":"...","message":"...","details":null}}`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Runtime/model configuration status |
| POST | `/api/datasets/inspect` | Validate and persist a multipart CSV |
| POST | `/api/projects` | Create a project for a stored dataset |
| GET | `/api/projects` | List persisted projects |
| POST | `/api/runs` | Queue an experiment |
| GET | `/api/runs` | List runs, optionally by `project_id` |
| GET | `/api/runs/{run_id}` | Poll run status |
| GET | `/api/runs/{run_id}/results` | Read metrics and visualization data |
| POST | `/api/runs/{run_id}/predict` | Predict a multipart feature-only CSV |
| GET | `/api/runs/{run_id}/download` | Download held-out predictions |

Interactive OpenAPI documentation is available at `/docs` while the server runs.

