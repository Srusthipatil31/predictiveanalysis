"""Deployment: REST API + prediction web page + results dashboard.

    python app.py      ->  http://127.0.0.1:5000        (prediction form)
                           http://127.0.0.1:5000/dashboard   (model results)
API:  POST /predict  {"mean radius": 17.9, ...}     GET /features     GET /health
"""
import json

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

from src import config as cfg
from src.predict import Predictor

app = Flask(__name__)
predictor = Predictor()


@app.route("/")
def index():
    return render_template("index.html", model_name=predictor.meta["model_name"])


@app.route("/health")
def health():
    return jsonify(status="ok", model=predictor.meta["model_name"], n_features=len(predictor.features))


@app.route("/features")
def features():
    m = predictor.meta
    return jsonify(features=m["feature_names"], medians=m["medians"], min=m["min"], max=m["max"],
                   examples=m["examples"])


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send a JSON object like {\"mean radius\": 17.9, ...}"), 400
    try:
        return jsonify(predictor.predict(payload))
    except ValueError as e:
        return jsonify(error=str(e)), 400


@app.route("/dashboard")
def dashboard():
    metrics = json.loads((cfg.OUTPUT_DIR / "metrics.json").read_text())
    comp = pd.read_csv(cfg.OUTPUT_DIR / "model_comparison.csv")
    imp = pd.read_csv(cfg.OUTPUT_DIR / "feature_importance.csv").head(8)
    cols = ["model", "cv_roc_auc", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
    return render_template("dashboard.html", m=metrics, comp=comp[cols].to_dict("records"),
                           imp=imp.to_dict("records"))


@app.route("/figures/<path:name>")
def figures(name):
    return send_from_directory(cfg.FIG_DIR, name)


if __name__ == "__main__":
    app.run(debug=False)
