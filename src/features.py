"""Image loading + feature extraction (OpenCV)."""
import cv2
import numpy as np

SIZE = 64


def decode(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read the image.")
    return img


def extract(img_bgr, mode="pixels"):
    """mode='pixels': flattened 64x64 RGB.  mode='stats': mean/max/min/mode/std per channel."""
    img = cv2.cvtColor(cv2.resize(img_bgr, (SIZE, SIZE)), cv2.COLOR_BGR2RGB)
    if mode == "pixels":
        return (img.astype(np.float32) / 255).ravel()
    f = []
    for c in range(3):
        ch = img[:, :, c].ravel()
        f += [ch.mean(), ch.max(), ch.min(), np.bincount(ch, minlength=256).argmax(), ch.std()]
    return np.array(f, np.float32) / 255
