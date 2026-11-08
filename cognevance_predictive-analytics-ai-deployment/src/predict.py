"""Load the saved pipeline and score new patients (used by the API, CLI and tests)."""
import json
import math

import joblib
import pandas as pd

from src import config as cfg


class Predictor:
    def __init__(self):
        self.model = joblib.load(cfg.MODEL_PATH)
        self.meta = json.loads(cfg.META_PATH.read_text())
        self.features = self.meta["feature_names"]

    def predict(self, values: dict) -> dict:
        """values: {feature name: number}. Missing features are filled with the training median."""
        clean, ignored = {}, []
        for k, v in (values or {}).items():
            if k not in self.features:
                ignored.append(k)
                continue
            if v is None or v == "":
                continue
            try:
                num = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"'{k}' must be a number, got {v!r}")
            if math.isnan(num) or math.isinf(num) or num < 0:
                raise ValueError(f"'{k}' must be a finite number >= 0")
            clean[k] = num
        if not clean:
            raise ValueError("Provide at least one valid feature value.")

        row = pd.DataFrame([{f: clean.get(f) for f in self.features}], columns=self.features, dtype=float)
        prob = float(self.model.predict_proba(row)[0, 1])
        return {
            "prediction": "Malignant" if prob >= 0.5 else "Benign",
            "probability_malignant": round(prob, 4),
            "risk_level": "high" if prob >= 0.8 else "medium" if prob >= 0.2 else "low",
            "features_provided": len(clean),
            "features_filled_with_median": len(self.features) - len(clean),
            "ignored_fields": ignored,
            "model": self.meta["model_name"],
            "disclaimer": "Educational demo only - not a medical diagnosis.",
        }
