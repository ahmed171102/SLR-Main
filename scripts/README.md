# Scripts - SLR System (Phase 1)

This directory will contain utility scripts for preparing the project for deployment.

## Planned Scripts

1.  `convert_models.py`: Python script to convert SavedModels (`.h5`) to TFLite format for mobile deployment.
    - **Must handle:** Custom layers (TemporalAttention) during conversion.
2.  `export_labels.py`: Export label encoders (from `.csv` or `.txt`) to JSON for frontend/mobile use.
3.  `copy_models.py`: Utility to aggregate all models from scattered subfolders into a centralized build directory.

See `../Deployment/docs/FULL_DEPLOYMENT_PLAN.md` for full instructions.
