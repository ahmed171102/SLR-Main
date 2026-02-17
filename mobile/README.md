# Mobile App - SLR System

This directory will contain the React Native (Expo) mobile application.

## Planned Structure (Phase 4)

- `App.tsx`: Main entry point.
- `src/screens/RecognizeScreen.tsx`: Camera and recognition logic.
- `src/services/tfliteModel.ts`: TFLite model inference (on-device).

## Key Requirements
- **TensorFlow Lite**: Models must be converted from `.h5` to `.tflite` using scripts in `../scripts/`.
- **MediaPipe**: Included in the mobile app or via API (if cloud-based). Ideally on-device for latency.

See `../Deployment/docs/FULL_DEPLOYMENT_PLAN.md` for full instructions.
