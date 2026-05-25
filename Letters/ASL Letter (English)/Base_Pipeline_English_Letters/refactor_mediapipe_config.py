import json
import re

notebook_path = r'm:\Term 10\Grad\SLR Main\Letters\ASL Letter (English)\Base_Pipeline_English_Letters\Mediapipe_Training.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

config_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Configuration & Hyperparameters\n",
        "All major paths, model parameters, and training settings are centralized here."
    ]
}

config_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- PATHS ---\n",
        "IMAGE_DATASET_DIR = r\"m:\\Term 10\\Grad\\SLR Main\\Letters\\ASL Letter (English)\\Merged_Dataset\\train\"\n",
        "CSV_SAVE_PATH     = \"asl_mediapipe_keypoints_dataset.csv\"\n",
        "MODEL_SAVE_PATH   = \"asl_mediapipe_mlp_model.h5\"\n",
        "BEST_MODEL_PATH   = \"asl_mediapipe_mlp_model_best.h5\"\n",
        "\n",
        "# --- MODEL ARCHITECTURE ---\n",
        "DENSE_1_UNITS = 256\n",
        "DENSE_2_UNITS = 128\n",
        "DENSE_3_UNITS = 64\n",
        "DROPOUT_1_RATE = 0.3\n",
        "DROPOUT_2_RATE = 0.25\n",
        "DROPOUT_3_RATE = 0.2\n",
        "L2_REGULARIZATION = 1e-4\n",
        "\n",
        "# --- TRAINING SETTINGS ---\n",
        "EPOCHS        = 20\n",
        "LEARNING_RATE = 0.001\n"
    ]
}

for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = "".join(cell.get('source', []))
        
        # Paths
        src = src.replace('CSV_PATH = "asl_mediapipe_keypoints_dataset.csv"', 'CSV_PATH = CSV_SAVE_PATH')
        src = re.sub(r"DATASET_DIR\s*=\s*r'[^']+'", "DATASET_DIR = IMAGE_DATASET_DIR", src)
        src = src.replace('pd.read_csv("asl_mediapipe_keypoints_dataset.csv")', 'pd.read_csv(CSV_SAVE_PATH)')
        src = src.replace('"asl_mediapipe_mlp_model.h5"', 'MODEL_SAVE_PATH')
        src = src.replace("'asl_mediapipe_mlp_model.h5'", 'MODEL_SAVE_PATH')
        src = src.replace("'asl_mediapipe_mlp_model_best.h5'", 'BEST_MODEL_PATH')
        
        # Architecture
        src = re.sub(r'Dense\(\s*256', 'Dense(\n            DENSE_1_UNITS', src)
        src = src.replace("l2(1e-4)", "l2(L2_REGULARIZATION)")
        src = src.replace("Dropout(0.3)", "Dropout(DROPOUT_1_RATE)")
        
        src = re.sub(r'Dense\(\s*128', 'Dense(\n            DENSE_2_UNITS', src)
        src = src.replace("Dropout(0.25)", "Dropout(DROPOUT_2_RATE)")
        
        src = re.sub(r'Dense\(\s*64', 'Dense(\n            DENSE_3_UNITS', src)
        src = src.replace("Dropout(0.2)", "Dropout(DROPOUT_3_RATE)")
        
        # Training Settings
        src = src.replace("learning_rate=0.001", "learning_rate=LEARNING_RATE")
        src = src.replace("epochs=20", "epochs=EPOCHS")
        
        cell['source'] = src.splitlines(keepends=True)

# Insert after the first code cell (which contains imports)
insert_idx = 0
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        insert_idx = i + 1
        break

nb['cells'].insert(insert_idx, config_code)
nb['cells'].insert(insert_idx, config_markdown)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    f.write('\n')

print("MediaPipe notebook refactored successfully!")
