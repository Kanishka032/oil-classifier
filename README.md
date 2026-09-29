# Oil Image Classifier (Edible vs Non-edible) - KNN + Flask, with real-time camera

Based on Kolakoti & Chandramouli, Fuel 349 (2023) 128618.

## Run (Windows, VS Code)
```
py -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/make_demo_data.py        # FAKE data, pipeline test only. Skip if you have real photos
python -m src.train --features pixels   # or --features stats
python app.py                           # http://127.0.0.1:5000 (Start camera = live prediction)
python -m src.realtime                  # optional OpenCV window, q to quit
python -m src.predict some_image.jpg    # optional single image
```
Real photos: dataset/edible/<oil>/*.jpg and dataset/non_edible/<oil>/*.jpg (30-50+ per oil).

## Honest accuracy
Training uses leave-oil-out cross-validation: each oil is held out entirely, so the score
is on oils the model has never seen. Expect well below 100% on real photos. Real-time
accuracy depends on lighting, distance and container: keep them the same as training.
Label mapping assumed: edible = coconut, cottonseed, groundnut, olive, rice bran, soybean,
sunflower; non-edible = algae, castor, jatropha, mahua, neem.

## Rejecting non-oil things (faces, room, hands)
KNN always picks the nearest class, so it needs to be told what "not oil" looks like.
1. Put 50+ photos of everything the camera may see besides oil (your face, hands, wall,
   table, empty beaker, water, bottle without oil) in `dataset/other/<any_name>/`, using the
   same camera and lighting. Then retrain: `python -m src.train`.
2. Even without this, images too far from all training photos are shown as "No oil detected"
   (distance threshold saved at training time).
