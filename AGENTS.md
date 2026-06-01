## Learned User Preferences

- Always preserve notebook structure (cells, headings, config, model architecture) when fixing or optimizing — never change the main architecture.
- Add comprehensive metrics and evaluation to every relevant cell, but without restructuring or rearranging the notebook.
- When creating diagnostic or utility notebooks, never modify the primary training notebook — create a new separate file instead.
- Each AI model is standalone with its own vocabulary — there is no shared vocabulary across languages or models.
- When the user asks to revert changes to a main notebook, always restore it and create a new notebook for the new concern rather than patching the original.
- Prefer Kaggle GPU (T4/P100/L4) for heavy training runs; local GPU is only 4 GB VRAM and is used for lighter experiments.
- For thesis writing, LaTeX is preferred over Word for structured academic documents.
- When suggesting training scope for the graduation project, the graduation vocabulary is 131 classes (100 Arabic words + 31 numbers), not the full 502-class KARSL dataset.
- Extend `Dataset Check.ipynb` by appending audit cells at the end when the user asks for new checks; preserve existing cells and structure (do not remove or reorder Cells 1–3).

## Learned Workspace Facts

- Project is a Sign Language Recognition (SLR) graduation project (FYP) — a full-stack application with four standalone AI models (Arabic/English words and letters).
- Arabic word dataset (KARSL-502): 502 classes at `E:\Downloads\Arabic Words Dataset`; all classes on disk in `train/` and `test/` via range subfolders (`0001-0070`, `0071-0170`, `0171-0190`, `0191-0300`, `0301-0502`).
- English word dataset: WLASL-2000 (2000 classes); notebooks in `Words/ASL Word (English)/`. How2Sign is used for continuous sign (CSLR) in separate notebooks.
- ArSL notebooks in `Words/ArSL Word (Arabic)/`: `ArSL_Word_Training_v2.ipynb` (full/partial 502-class extraction), `ArSL_Word_Training_Kagglev2.ipynb` (Kaggle), `ArSL_Word_Training_CustomWords.ipynb` (131-class graduation; BiLSTM + MHA; Cell 3b targeted extraction; `REQUIRE_FULL_VOCABULARY` and `BLOCK_HOLDOUT_SPLIT=True`).
- `Dataset Check.ipynb`: Cells 1–3 scan dataset integrity; Cell 4 audits 131 graduation sign_ids vs disk and NPZ (`arsl_graduation_sequences.npz` → full → partial); Cell 5 audits graduation training artifacts (memory-safe — avoid loading TensorFlow after large NPZs).
- KARSL-502 taxonomy: sign_id 1–31 numbers, 32–70 letters, 71–502 words; graduation uses words + numbers only. `KARSL-502_GraduationSelection.csv` defines the 131 sign_ids; run `_build_graduation_from_master.py` to refresh `KARSL-502_BasicWords.csv`.
- `arsl_graduation_sequences.npz` is the canonical `SOURCE_NPZ` for graduation (6616×48×258, 131/131 classes, 258-dim MediaPipe face+hands); built via Cell 3b in `ArSL_Word_Training_CustomWords.ipynb` (~3–4 h CPU). Legacy `arsl_word_sequences_v2_partial.npz` covers class_id 1–200 only (~64/131 classes).
- Graduation model artifacts in `Words/ArSL Word (Arabic)/`: `arsl_custom_best.h5`, `arsl_custom_scaler.npz`, `arsl_custom_classes.csv`, plots, and `arsl_custom_eval_metrics.csv`; confirmed thesis metrics: **96.64% top-1, 96.66% macro-F1, 98.40% top-5**, 131 classes, 1189 test samples, block_holdout split.
- `arsl_custom_subset.npz` `y` array stores raw KARSL class IDs (not 0-indexed model indices); `ArSL_Word_Diagnostics.ipynb` uses a `karsl_to_model` dict (built from `arsl_custom_classes.csv`) to remap before all metric computations.
- `ArSL_Word_Live_Test.ipynb` Cell 5 uses a 1280×720 split-screen (`np.hstack`): left 640×720 = camera feed + HUD (MediaPipe landmarks drawn on resized region), right 640×720 = all 131-class probability bars sorted by confidence; `all_probs` list (all 131 sorted desc) replaces the old `top3`; bar colours encode rank (top-1 = confidence colour, 2-3 = bright green, 4-10 = teal, 11+ = dim).
- Thesis LaTeX scaffold: `Current Thesis/latex/`; template PDF: `Current Thesis/FYP Template Final - Individual Report.pdf`.
- `SLR_Diagnostics.ipynb` at repo root is shared — do not modify it for language-specific work; copy or use `Dataset Check.ipynb` instead.
