import json

file_path = r'm:\Term 10\Grad\SLR Main\Letters\Merger Notebook new approach\Trials of the model\Letters\Trial #2\mediapipe-v2-2-train-only.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []
code_cell_counter = 1

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        
        label = "Code Block"
        desc = ""
        
        if 'pip uninstall' in src or 'pip install' in src:
            label = "Environment Setup"
            desc = "*Installing and updating MediaPipe and Protobuf dependencies.*"
        elif 'import cv2' in src and 'tensorflow' in src:
            label = "Library Imports"
            desc = "*Importing essential Python libraries (OpenCV, MediaPipe, TensorFlow, Pandas).*"
        elif 'IMAGE_DATASET_DIR =' in src or 'DENSE_1_UNITS' in src:
            label = "Paths & Hyperparameters Configuration"
            desc = "*Defining dataset paths, model units, dropout rates, and training settings.*"
        elif 'tf.config.list_physical_devices' in src and 'GPU DETECTION' in src:
            label = "GPU Detection & Setup"
            desc = "*Checking for available GPUs and configuring memory growth.*"
        elif 'GPU MEMORY MANAGEMENT TIPS' in src:
            label = "GPU Memory Guidelines"
            desc = "*Tips for optimizing VRAM usage during training.*"
        elif 'hands.process' in src or 'EXTRACTING MEDIAPIPE' in src:
            label = "MediaPipe Feature Extraction"
            desc = "*Processing the raw images, extracting hand landmarks, and saving to CSV.*"
        elif 'train_test_split' in src and 'LabelEncoder' in src:
            label = "Data Splitting (Train / Val / Test)"
            desc = "*Loading the CSV, balancing classes, and performing the 3-way split.*"
        elif 'tf.data.Dataset.from_tensor_slices' in src:
            label = "TensorFlow Data Pipelines"
            desc = "*Creating high-performance tf.data pipelines for efficient training.*"
        elif 'Sequential' in src and 'Dense' in src:
            label = "Model Architecture (MLP)"
            desc = "*Building the Multi-Layer Perceptron neural network.*"
        elif 'model.compile' in src or 'ModelCheckpoint' in src:
            label = "Model Compilation & Callbacks"
            desc = "*Setting up the optimizer, loss function, EarlyStopping, and Checkpoints.*"
        elif 'model.fit' in src:
            label = "Model Training"
            desc = "*Executing the training loop over the dataset.*"
        elif 'model.evaluate' in src and 'history' not in src:
            label = "Model Evaluation"
            desc = "*Testing the model on the unseen Test set.*"
        elif 'matplotlib' in src or 'plot' in src:
            label = "Training History Visualization"
            desc = "*Plotting Accuracy and Loss curves.*"
        elif 'confusion_matrix' in src or 'classification_report' in src:
            label = "Performance Metrics"
            desc = "*Generating Confusion Matrix and Classification Report.*"
        elif 'cv2.VideoCapture' in src:
            label = "Real-Time Inference (Webcam)"
            desc = "*Running the trained model on live webcam feed.*"
        elif 'FINAL PROJECT REPORT' in src:
            label = "Final Report Summary"
            desc = "*Printing the final model specifications and performance conclusion.*"
        else:
            if len(src.strip()) > 0:
                label = "Utility / Execution"
                desc = "*Additional helper code.*"
            else:
                label = "Empty Cell"
                desc = ""
                
        if label != "Empty Cell":
            new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"### CODE CELL {code_cell_counter}: {label}\n", f"{desc}"]
            })
            
        code_cell_counter += 1
        new_cells.append(cell)
    else:
        new_cells.append(cell)

nb['cells'] = new_cells

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Labeled all code cells successfully.")
