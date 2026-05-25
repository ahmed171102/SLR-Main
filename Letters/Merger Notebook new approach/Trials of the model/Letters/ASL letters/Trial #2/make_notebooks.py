import json
import os
import shutil

dir_path = r'm:\Term 10\Grad\SLR Main\Letters\Merger Notebook new approach\Trials of the model\Letters\Trial #2'

# 1. mediapipe-v2-2.ipynb
shutil.copy(
    os.path.join(dir_path, 'mediapipe-v2-2.ipynb'),
    os.path.join(dir_path, 'mediapipe-v2-2-train-only.ipynb')
)

# 2. mobile-net-v1-2.ipynb
with open(os.path.join(dir_path, 'mobile-net-v1-2.ipynb'), 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Modify Cell 14 (Paths)
        if 'TEST_DATASET_DIR' in source and 'TRAIN_DATASET_DIR' in source:
            new_source = []
            for line in cell['source']:
                if 'TEST_DATASET_DIR' not in line:
                    new_source.append(line)
            cell['source'] = new_source
            new_cells.append(cell)
            continue
            
        # Insert DataFrame Split Cell before the ImageDataGenerator cell
        if 'train_datagen = ImageDataGenerator(' in source and 'preprocessing_function=preprocess_input' in source:
            df_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import os\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "\n",
                    "# BEST PRACTICE: 3-Way Split (Train/Val/Test) using DataFrames\n",
                    "filepaths = []\n",
                    "labels = []\n",
                    "for cls in sorted(os.listdir(dataset_dir)):\n",
                    "    cls_dir = os.path.join(dataset_dir, cls)\n",
                    "    if os.path.isdir(cls_dir):\n",
                    "        for img in os.listdir(cls_dir):\n",
                    "            if img.lower().endswith(('.jpg', '.jpeg', '.png')):\n",
                    "                filepaths.append(os.path.join(cls_dir, img))\n",
                    "                labels.append(cls)\n",
                    "\n",
                    "df = pd.DataFrame({'filepath': filepaths, 'label': labels})\n",
                    "\n",
                    "# Remove classes with fewer than 2 samples to fix stratification\n",
                    "class_counts = df['label'].value_counts()\n",
                    "valid_classes = class_counts[class_counts >= 2].index\n",
                    "df = df[df['label'].isin(valid_classes)]\n",
                    "\n",
                    "print(f\"Total valid images found: {len(df)}\")\n",
                    "\n",
                    "# Split: 80% Train, 10% Val, 10% Test\n",
                    "train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)\n",
                    "val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)\n",
                    "\n",
                    "print(f\"Training samples: {len(train_df)}\")\n",
                    "print(f\"Validation samples: {len(val_df)}\")\n",
                    "print(f\"Test samples: {len(test_df)}\")\n"
                ]
            }
            new_cells.append(df_cell)
            
            # Now modify the ImageDataGenerator cell to use flow_from_dataframe
            cell['source'] = [
                "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
                "from tensorflow.keras.applications.mobilenet_v2 import preprocess_input\n",
                "\n",
                "# Data generators\n",
                "train_datagen = ImageDataGenerator(\n",
                "    preprocessing_function=preprocess_input,\n",
                "    rotation_range=20,\n",
                "    width_shift_range=0.2,\n",
                "    height_shift_range=0.2,\n",
                "    shear_range=0.2,\n",
                "    zoom_range=0.2,\n",
                "    horizontal_flip=False\n",
                ")\n",
                "\n",
                "val_datagen = ImageDataGenerator(\n",
                "    preprocessing_function=preprocess_input\n",
                ")\n",
                "\n",
                "# Train & validation generators using DataFrames\n",
                "train_generator = train_datagen.flow_from_dataframe(\n",
                "    dataframe=train_df,\n",
                "    x_col='filepath',\n",
                "    y_col='label',\n",
                "    target_size=(IMG_SIZE, IMG_SIZE),\n",
                "    batch_size=BATCH_SIZE,\n",
                "    class_mode='categorical',\n",
                "    seed=42\n",
                ")\n",
                "\n",
                "val_generator = val_datagen.flow_from_dataframe(\n",
                "    dataframe=val_df,\n",
                "    x_col='filepath',\n",
                "    y_col='label',\n",
                "    target_size=(IMG_SIZE, IMG_SIZE),\n",
                "    batch_size=BATCH_SIZE,\n",
                "    class_mode='categorical',\n",
                "    seed=42,\n",
                "    shuffle=False\n",
                ")\n",
                "\n",
                "print(\"Class labels:\", train_generator.class_indices)\n"
            ]
            new_cells.append(cell)
            continue
            
        # Modify Cell 36 (Restructuring test data)
        if 'original_test_dir = TEST_DATASET_DIR' in source:
            cell['source'] = [
                "# This cell was intentionally disabled because we are generating the test part directly from the training dataset.\n",
                "# original_test_dir = TEST_DATASET_DIR\n",
                "# ...\n",
                "print('Test data structuring skipped. Using test dataframe instead.')\n"
            ]
            new_cells.append(cell)
            continue
            
        # Modify Cell 37 (Test Generator)
        if 'test_datagen.flow_from_directory' in source and 'test_folder' in source:
            cell['source'] = [
                "from tensorflow.keras.applications.mobilenet_v2 import preprocess_input\n",
                "\n",
                "test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)\n",
                "\n",
                "test_generator = test_datagen.flow_from_dataframe(\n",
                "    dataframe=test_df,\n",
                "    x_col='filepath',\n",
                "    y_col='label',\n",
                "    target_size=(IMG_SIZE, IMG_SIZE),\n",
                "    batch_size=BATCH_SIZE,\n",
                "    class_mode='categorical',\n",
                "    shuffle=False,\n",
                "    seed=42\n",
                ")\n",
                "\n",
                "print(f\"Loaded test images from test dataframe\")\n"
            ]
            new_cells.append(cell)
            continue

    # Add unmodified cells
    new_cells.append(cell)

nb['cells'] = new_cells

with open(os.path.join(dir_path, 'mobile-net-v1-2-train-only.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Done updating notebooks to best practice.")
