# Initial audit

Date: 2026-09-02

## Verified local environment

- Windows host with `C:\Users\ymbei\anaconda3\envs\limix\python.exe`.
- PyTorch `2.7.1+cu128` reports CUDA available.
- NVIDIA Quadro P620 is visible with 4 GB VRAM.
- LimiX-2M checkpoint exists at
  `C:\Users\ymbei\Desktop\Limix\LimiX-main\cache\LimiX-2M.ckpt`.
- The upstream code and checkpoint are local runtime dependencies and are not copied into this repository.

## Upstream API confirmed

The supported integration point is `inference.predictor.LimiXPredictor`:

```python
predictor = LimiXPredictor(
    device=torch.device("cuda"),
    model_path=".../LimiX-2M.ckpt",
    inference_config=".../config/cls_default_noretrieval.json",
)
prediction = predictor.predict(x_train, y_train, x_test, task_type="Classification")
```

Classification returns per-class probabilities. Regression returns a tensor and
requires target normalization and inverse transformation in the application layer.

## Existing application findings

The existing `app.py` is a Streamlit proof of concept. A real classification run
completed, but the implementation is not a production workbench:

- UI, preprocessing, inference, metrics, and downloads are coupled in one module.
- Exceptions are rendered with `st.exception`, exposing tracebacks to end users.
- The classification path refits a label encoder on test labels.
- The upstream regression command aborts when CUDA is unavailable.
- There is no API, persistent run history, batch inference contract, or job status.
- Upstream examples still default to LimiX-16M and retrieval-oriented configs.

## Implementation boundary

This repository is an independent application. It imports the upstream LimiX
source tree at runtime using `LIMIX_SOURCE_DIR`, reads the local LimiX-2M checkpoint
from `LIMIX_MODEL_PATH`, and uses only no-retrieval configs. LimiX core source and
weights remain unchanged and outside Git.

