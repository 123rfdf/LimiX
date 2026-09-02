from __future__ import annotations

from pathlib import Path

import numpy as np
from app.services.limix_adapter import AdapterSettings, LimiXAdapter


def test_regression_inverse_scaling_does_not_overflow_float16() -> None:
    adapter = LimiXAdapter(
        AdapterSettings(
            source_dir=Path("unused"),
            model_path=Path("LimiX-2M.ckpt"),
            classification_config=Path("classification.json"),
            regression_config=Path("regression.json"),
        )
    )
    adapter._predict_with_fallback = (  # type: ignore[method-assign]
        lambda *_args: (np.array([0.5, -0.5], dtype=np.float16), 0.01, "cuda")
    )
    output = adapter.regress(
        np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
        np.array([100_000, 200_000, 300_000, 400_000], dtype=np.float32),
        np.array([[1.5], [2.5]], dtype=np.float32),
    )
    assert np.isfinite(output.predictions).all()
    assert output.predictions.dtype == np.float64

