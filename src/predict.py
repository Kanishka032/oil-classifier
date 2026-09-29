"""Prediction helpers.  CLI:  python -m src.predict image.jpg"""
import sys
from pathlib import Path

import cv2
import joblib

from .features import decode, extract

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "knn_oil.joblib"
NO_OIL = "no oil detected"


def load_bundle(path=MODEL_PATH):
    return joblib.load(path)


def predict_frame(bundle, img_bgr):
    x = extract(img_bgr, bundle["mode"]).reshape(1, -1)
    model = bundle["model"]
    proba = model.predict_proba(x)[0]
    classes = list(model.classes_)
    i = int(proba.argmax())
    label, conf = classes[i], round(float(proba[i]), 3)
    # Guard 1: image is too far from every training photo -> not an oil (face, room, hand...)
    dist = float(model[-1].kneighbors(model[:-1].transform(x), n_neighbors=1)[0][0][0])
    too_far = bundle.get("max_dist") is not None and dist > bundle["max_dist"]
    # Guard 2: the model was trained with an "other" class (non-oil photos)
    if too_far or label == "other":
        label, conf = NO_OIL, 0.0
    return {"label": label, "confidence": conf, "distance": round(dist, 2),
            "probabilities": {c: round(float(p), 3) for c, p in zip(classes, proba)}}


def predict_bytes(bundle, data: bytes):
    return predict_frame(bundle, decode(data))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.predict image.jpg")
    img = cv2.imread(sys.argv[1])
    if img is None:
        raise SystemExit("Could not read image.")
    print(predict_frame(load_bundle(), img))
