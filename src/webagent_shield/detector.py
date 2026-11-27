"""Optional classical detector used at the inspection boundary."""
import os
import math
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4)
def _load_model(path: str):
    import joblib

    return joblib.load(path)


def status() -> str:
    configured_path = os.getenv("SHIELD_MODEL_PATH", "").strip()
    if not configured_path:
        return "not_trained"
    try:
        _load_model(str(Path(configured_path).resolve()))
    except Exception:
        return "error"
    return "trained"


def scan(content: str) -> dict:
    configured_path = os.getenv("SHIELD_MODEL_PATH", "").strip()
    if not configured_path:
        return {"detector_status": "not_trained", "findings": []}
    path = Path(configured_path)
    try:
        model = _load_model(str(path.resolve()))
        classifier = model.named_steps["classifier"]
        if hasattr(classifier, "decision_function"):
            margin = float(model.decision_function([content])[0])
        else:
            margin = float(model.predict_proba([content])[0][1] - 0.5)
        threshold = float(os.getenv("SHIELD_DECISION_THRESHOLD", "0.5"))
        if not math.isfinite(threshold):
            raise ValueError("Detector threshold must be finite")
        prediction = int(margin >= threshold)
    except Exception as error:
        return {
            "detector_status": "error",
            "findings": [{"type": "detector_error", "scope": "page",
                          "detail": type(error).__name__}],
        }
    findings = []
    if prediction == 1:
        findings.append({"type": "prompt_injection", "scope": "page",
                         "model": path.name, "decision_margin": margin,
                         "threshold": threshold})
    return {"detector_status": "trained", "findings": findings}
