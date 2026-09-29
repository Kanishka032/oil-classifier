"""Train KNN.  python -m src.train --features pixels|stats"""
import argparse
import json
from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import extract

ROOT = Path(__file__).resolve().parent.parent
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load(mode):
    X, y, g = [], [], []
    for cls in ("edible", "non_edible", "other"):  # "other" = non-oil photos
        for oil in sorted((ROOT / "dataset" / cls).glob("*")):
            for p in oil.glob("*"):
                img = cv2.imread(str(p)) if p.suffix.lower() in EXTS else None
                if img is not None:
                    X.append(extract(img, mode)); y.append(cls); g.append(oil.name)
    return np.array(X), np.array(y), np.array(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["pixels", "stats"], default="pixels")
    mode = ap.parse_args().features
    X, y, g = load(mode)
    if len(set(y)) < 2:
        raise SystemExit("Need images in dataset/edible/<oil>/ and dataset/non_edible/<oil>/")
    n_oils = len(set(g))
    print(f"{len(X)} images, {n_oils} oils, features={mode}")

    # Group CV: every oil is held out together, so test oils are never seen in training.
    cv = GroupKFold(n_splits=min(5, n_oils))
    pipe = Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier())])
    grid = {"knn__n_neighbors": [1, 3, 5, 7, 9, 11], "knn__weights": ["uniform", "distance"],
            "knn__metric": ["euclidean", "manhattan"]}
    gs = GridSearchCV(pipe, grid, cv=cv, n_jobs=-1).fit(X, y, groups=g)
    pred = cross_val_predict(gs.best_estimator_, X, y, cv=cv, groups=g)
    acc = float((pred == y).mean())
    labels = sorted(set(y))
    cm = confusion_matrix(y, pred, labels=labels)
    print("Best params:", gs.best_params_)
    print(f"Leave-oil-out CV accuracy: {acc:.3f}")
    print(classification_report(y, pred))

    out = ROOT / "static" / "analytics"
    out.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues")
    plt.title("Confusion matrix (leave-oil-out CV)")
    plt.savefig(out / "confusion.png", dpi=110, bbox_inches="tight")
    (out / "metrics.json").write_text(json.dumps({
        "features": mode, "images": len(X), "oils": n_oils, "accuracy": round(acc, 4),
        "best_params": {k.split("__")[1]: v for k, v in gs.best_params_.items()},
        "report": classification_report(y, pred, output_dict=True)}, indent=2, default=float))

    (ROOT / "models").mkdir(exist_ok=True)
    final = gs.best_estimator_.fit(X, y)
    # Rejection threshold: how far a real oil photo normally is from its nearest neighbour
    oil = y != "other"
    d = final[-1].kneighbors(final[:-1].transform(X[oil]), n_neighbors=2)[0][:, 1]
    max_dist = float(np.percentile(d, 99) * 1.25)
    print(f"Rejection distance threshold: {max_dist:.2f}")
    joblib.dump({"model": final, "mode": mode, "max_dist": max_dist}, ROOT / "models" / "knn_oil.joblib")
    print("Saved models/knn_oil.joblib")


if __name__ == "__main__":
    main()
