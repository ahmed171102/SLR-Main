# Backend - SLR System

This directory will contain the Python (FastAPI) backend for serving sign language predictions.

## Planned Structure (Phase 2)

- `app/main.py`: Main entry point.
- `app/models/`: Model loading and inference (`.h5` files, temporal attention).
- `app/routes/`: REST API and WebSocket endpoints.
- `requirements.txt`: Backend dependencies.
- `Dockerfile`: Container configuration.

## Key Models Required
- **Letter Model (MLP)**: Trained with MediaPipe landmarks.
- **Word Model (BiLSTM)**: Temporal sequence model.
- **MobileNetV2**: Transfer learning model (optional).

See `../Deployment/docs/FULL_DEPLOYMENT_PLAN.md` for full instructions.
