# Face Landmarker Model

This directory stores the official Google MediaPipe Face Landmarker model bundle (`face_landmarker.task`).

## Model Details
- **Model**: MediaPipe Face Landmarker (Float16)
- **Official Source**: Google MediaPipe Model Storage
- **Direct Official URL**: [https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)
- **Expected Landmarks**: 478 3D landmarks (468 facial mesh landmarks + 10 iris landmarks)

## Automatic Download
FaceScope automatically downloads this model if missing on first launch via `src.model_loader.download_model_if_missing()`.

To manually download the model:
```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task', 'models/face_landmarker.task')"
```
