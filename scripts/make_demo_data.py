"""SYNTHETIC images for pipeline testing only (not real oil photos)."""
from pathlib import Path

import cv2
import numpy as np

rng = np.random.default_rng(0)
ROOT = Path(__file__).resolve().parent.parent / "dataset"
BGR = {  # rough colours; groups overlap on purpose
    "edible": {"coconut": (200, 235, 245), "cottonseed": (60, 170, 225), "groundnut": (70, 190, 230),
               "olive": (60, 170, 170), "rice_bran": (80, 180, 225), "soybean": (75, 185, 235),
               "sunflower": (50, 200, 240)},
    "non_edible": {"algae": (40, 120, 130), "castor": (170, 220, 235), "jatropha": (60, 150, 200),
                   "mahua": (50, 130, 190), "neem": (30, 100, 150)},
    "other": {"skin": (140, 170, 215), "wall": (220, 225, 230), "dark_room": (30, 30, 35),
              "blue_shirt": (160, 90, 50)},
}
for cls, oils in BGR.items():
    for oil, col in oils.items():
        d = ROOT / cls / oil
        d.mkdir(parents=True, exist_ok=True)
        for i in range(20):
            base = np.array(col, float) * rng.uniform(0.8, 1.2) + rng.normal(0, 12, 3)
            img = np.clip(base + rng.normal(0, 10, (128, 128, 3)), 0, 255).astype(np.uint8)
            cv2.imwrite(str(d / f"{i}.jpg"), img)
print("Demo data written to dataset/")
