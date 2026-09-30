"""Check original evaluator fingerprints, optionally load and score a fixed peptide."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(root: Path, smoke: bool = False) -> dict:
    manifest = json.loads((root / "shared-evaluator/models/ASSET_MANIFEST.json").read_text())
    for asset in manifest["models"]:
        path = root / asset["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_size != asset["bytes"] or digest != asset["sha256"]:
            raise ValueError(f"Evaluator fingerprint mismatch: {asset['path']}")
    report = {"verified_models": len(manifest["models"]), "loaded_models": False}
    if smoke:
        # Only deserialize after verifying every model against the tracked manifest.
        import numpy as np
        import sklearn
        if sklearn.__version__ != manifest["runtime"]["scikit-learn"]:
            raise ValueError("Use the locked shared-evaluator environment for model loading")
        sys.path.insert(0, str(root / "shared-evaluator/src"))
        from evaluator.activity_safety_models import ActivitySafetyEvaluator
        evaluator = ActivitySafetyEvaluator(root / "shared-evaluator/models")
        for model in (evaluator.amp_model, evaluator.hemo_model):
            if model.n_features_in_ != 27 or len(model.estimators_) != 50:
                raise ValueError("Unexpected forest dimensions")
            if list(model.classes_) != [0, 1]:
                raise ValueError("Unexpected classifier labels")
        sequence = "KWKLFKKIGAVLKVL"
        prediction = evaluator.predict(sequence)
        if not np.isfinite(list(prediction.values())).all():
            raise ValueError("Non-finite evaluator prediction")
        if any(not 0 <= value <= 1 for value in prediction.values()):
            raise ValueError("Evaluator result outside expected range")
        report.update(loaded_models=True, sequence=sequence, prediction=prediction,
                      runtime={"scikit-learn": sklearn.__version__, "numpy": np.__version__})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify(args.root.resolve(), args.smoke)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
