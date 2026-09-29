"""Live webcam prediction.  python -m src.realtime   (q to quit)"""
from collections import Counter, deque

import cv2

from .predict import load_bundle, predict_frame

bundle = load_bundle()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam.")
recent, n, text = deque(maxlen=5), 0, "starting..."
while True:
    ok, frame = cap.read()
    if not ok:
        break
    n += 1
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = w // 4, h // 4, 3 * w // 4, 3 * h // 4
    if n % 5 == 0:
        r = predict_frame(bundle, frame[y1:y2, x1:x2])
        recent.append(r["label"])
        top = Counter(recent).most_common(1)[0][0]  # smoothing over last 5 predictions
        text = top if top == "no oil detected" else f"{top} ({r['confidence']*100:.0f}%)"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Oil classifier - q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
