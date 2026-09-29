"""Flask backend.  Run:  python app.py  ->  http://127.0.0.1:5000"""
import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.predict import load_bundle, predict_bytes

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "knn_oil.joblib"
METRICS_PATH = ROOT / "static" / "analytics" / "metrics.json"
ALLOWED = {"jpg", "jpeg", "png", "bmp", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None and MODEL_PATH.exists():
        _bundle = load_bundle(MODEL_PATH)
    return _bundle


@app.route("/")
def index():
    return render_template("index.html", ready=get_bundle() is not None)


@app.route("/predict", methods=["POST"])
def predict():
    bundle = get_bundle()
    if bundle is None:
        return jsonify(error="Model not trained yet. Run: python -m src.train"), 503
    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify(error="Choose an image first."), 400
    if file.filename.rsplit(".", 1)[-1].lower() not in ALLOWED:
        return jsonify(error="Use a JPG, PNG, BMP or WEBP image."), 400
    try:
        return jsonify(predict_bytes(bundle, file.read()))
    except ValueError as e:
        return jsonify(error=str(e)), 400


@app.route("/analytics")
def analytics():
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None
    return render_template("analytics.html", m=metrics)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image is larger than 8 MB."), 413


if __name__ == "__main__":
    app.run(debug=True)
