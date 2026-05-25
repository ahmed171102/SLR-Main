import json

notebook_path = r'm:\Term 10\Grad\SLR Main\Letters\ASL Letter (English)\Base_Pipeline_English_Letters\MobileNetV2_Training.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

config_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Configuration & Hyperparameters\n",
        "All major paths, model parameters, and training settings are centralized here so you can easily modify them without hunting through the code."
    ]
}

config_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- PATHS ---\n",
        "TRAIN_DATASET_DIR = r\"m:\\Term 10\\Grad\\SLR Main\\Letters\\ASL Letter (English)\\Merged_Dataset\\train\"\n",
        "TEST_DATASET_DIR  = r\"m:\\Term 10\\Grad\\SLR Main\\Letters\\ASL Letter (English)\\Merged_Dataset\\test\"\n",
        "MODEL_SAVE_PATH   = \"sign_language_model_MobileNetV2.h5\"\n",
        "\n",
        "# --- IMAGE & BATCH SETTINGS ---\n",
        "IMG_SIZE   = 128 \n",
        "BATCH_SIZE = 32\n",
        "\n",
        "# --- MODEL ARCHITECTURE ---\n",
        "DENSE_UNITS  = 256\n",
        "DROPOUT_RATE = 0.3\n",
        "\n",
        "# --- TRAINING SETTINGS ---\n",
        "INITIAL_EPOCHS      = 5\n",
        "FINETUNE_EPOCHS     = 10\n",
        "FINETUNE_LEARN_RATE = 1e-4\n"
    ]
}

old_mac_path = "'/Users/js/Desktop/Sign Recognition Application/Sign_to_Sentence Project/Asl_Sign_Data/asl_alphabet_train/asl_alphabet_train'"
old_mac_test = "'/Users/js/Desktop/Sign Recognition Application/Sign_to_Sentence Project/Asl_Sign_Data/asl_alphabet_test/asl_alphabet_test' # UPDATE THIS TO YOUR MERGED TEST FOLDER"
old_mac_test2 = "'/Users/js/Desktop/Sign Recognition Application/Sign_to_Sentence Project/Asl_Sign_Data/asl_alphabet_test/asl_alphabet_test'"

first_usage_idx = -1

for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = "".join(cell.get('source', []))
        
        # We want to find the first cell where dataset_dir is used to insert the config block before it
        if "dataset_dir =" in src and first_usage_idx == -1:
            first_usage_idx = i

        # Paths
        src = src.replace(old_mac_path, "TRAIN_DATASET_DIR")
        src = src.replace(old_mac_test, "TEST_DATASET_DIR")
        src = src.replace(old_mac_test2, "TEST_DATASET_DIR")
        src = src.replace('"sign_language_model_MobileNetV2.h5"', "MODEL_SAVE_PATH")
        
        # Image / Batch size
        src = src.replace("IMG_SIZE = 128 \n", "")
        src = src.replace("BATCH_SIZE = 32\n", "")
        
        # Model architecture
        src = src.replace("Dense(256,", "Dense(DENSE_UNITS,")
        src = src.replace("Dropout(0.3)", "Dropout(DROPOUT_RATE)")
            
        # Training
        src = src.replace("epochs=5 ", "epochs=INITIAL_EPOCHS ")
        src = src.replace("epochs=5\n", "epochs=INITIAL_EPOCHS\n")
        src = src.replace("epochs=10", "epochs=FINETUNE_EPOCHS")
        src = src.replace("learning_rate=1e-4", "learning_rate=FINETUNE_LEARN_RATE")
            
        cell['source'] = src.splitlines(keepends=True)

if first_usage_idx != -1:
    nb['cells'].insert(first_usage_idx, config_code)
    nb['cells'].insert(first_usage_idx, config_markdown)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    f.write('\n')

print("Config block added successfully!")
