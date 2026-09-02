from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.limix_adapter import AdapterSettings, LimiXAdapter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real local LimiX-2M smoke tests.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    adapter = LimiXAdapter(
        AdapterSettings(
            source_dir=args.source,
            model_path=args.model,
            classification_config=args.source / "config" / "cls_default_noretrieval.json",
            regression_config=args.source / "config" / "reg_default_noretrieval.json",
            preferred_device=args.device,
        )
    )

    x_cls, y_cls = load_breast_cancer(return_X_y=True)
    x_cls, _, y_cls, _ = train_test_split(
        x_cls, y_cls, train_size=120, random_state=7, stratify=y_cls
    )
    x_cls_train, x_cls_test, y_cls_train, y_cls_test = train_test_split(
        x_cls, y_cls, test_size=24, random_state=7, stratify=y_cls
    )
    cls_output = adapter.classify(x_cls_train, y_cls_train, x_cls_test)

    x_reg, y_reg = load_diabetes(return_X_y=True)
    x_reg, _, y_reg, _ = train_test_split(x_reg, y_reg, train_size=120, random_state=7)
    x_reg_train, x_reg_test, y_reg_train, y_reg_test = train_test_split(
        x_reg, y_reg, test_size=24, random_state=7
    )
    reg_output = adapter.regress(x_reg_train, y_reg_train, x_reg_test)

    print(
        json.dumps(
            {
                "classification": {
                    "accuracy": accuracy_score(y_cls_test, cls_output.predictions),
                    "seconds": cls_output.inference_seconds,
                    "device": cls_output.device,
                    "probability_shape": list(cls_output.probabilities.shape),
                },
                "regression": {
                    "r2": r2_score(y_reg_test, reg_output.predictions),
                    "seconds": reg_output.inference_seconds,
                    "device": reg_output.device,
                    "prediction_count": len(reg_output.predictions),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
