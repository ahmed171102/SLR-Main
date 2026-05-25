import json
import os

notebook_path = r'm:\Term 10\Grad\SLR Main\Letters\ASL Letter (English)\Base_Pipeline_English_Letters\MobileNetV2_Training.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Modification 2
        if 'class_labels = {' in source and '"A"' in source and '"space"' in source:
            cell['source'] = [
                "# Generate class labels dynamically based on alphabetical folder sorting\n",
                "class_labels = {v: k for k, v in train_generator.class_indices.items()}\n",
                "print(f\"Dynamically generated class labels: {class_labels}\")\n"
            ]
            
        # Modification 3 - Block 1 (test folder loading)
        elif 'test_folder =' in source and 'test_images = []' in source:
            cell['source'] = [
                "IMG_SIZE = 128 \n",
                "BATCH_SIZE = 32\n",
                "test_folder = '/Users/js/Desktop/Sign Recognition Application/Sign_to_Sentence Project/Asl_Sign_Data/asl_alphabet_test/asl_alphabet_test' # UPDATE THIS TO YOUR MERGED TEST FOLDER\n",
                "\n",
                "test_datagen = ImageDataGenerator(rescale=1./255)\n",
                "test_generator = test_datagen.flow_from_directory(\n",
                "    test_folder,\n",
                "    target_size=(IMG_SIZE, IMG_SIZE),\n",
                "    batch_size=BATCH_SIZE,\n",
                "    class_mode=\"categorical\",\n",
                "    shuffle=False # Very important for evaluation to match predictions with true labels\n",
                ")\n",
                "\n",
                "print(f\"Loaded test images from {test_folder}\")\n"
            ]
            
        # Modification 3 - Block 2 (predictions)
        elif 'predictions = model.predict(test_images)' in source:
            cell['source'] = [
                "# Get model predictions\n",
                "predictions = model.predict(test_generator)\n",
                "\n",
                "# Convert probabilities to class labels\n",
                "predicted_classes = np.argmax(predictions, axis=1)\n",
                "true_classes = test_generator.classes\n",
                "\n",
                "# Print first 10 predictions as a sample\n",
                "filenames = test_generator.filenames\n",
                "for i in range(min(10, len(filenames))):\n",
                "    pred_label = class_labels[predicted_classes[i]]\n",
                "    true_label = class_labels[true_classes[i]]\n",
                "    print(f\"Image: {filenames[i]} → Predicted as: {pred_label} (True: {true_label})\")\n"
            ]
            
        # Modification 3 - Block 3 (accuracy)
        elif 'true_labels = [img_name.split("_")[0] for img_name in image_names]' in source:
            cell['source'] = [
                "correct = np.sum(predicted_classes == true_classes)\n",
                "accuracy = (correct / len(true_classes)) * 100\n",
                "\n",
                "print(f\"Test Accuracy: {accuracy:.2f}%\")\n"
            ]
            
        # Modification 3 - Block 4 (confusion matrix)
        elif 'cm = confusion_matrix(true_labels, [class_labels[i] for i in predicted_classes]' in source:
            cell['source'] = [
                "import seaborn as sns\n",
                "from sklearn.metrics import confusion_matrix\n",
                "\n",
                "label_names = [class_labels[i] for i in range(len(class_labels))]\n",
                "cm = confusion_matrix(true_classes, predicted_classes, labels=list(range(len(class_labels))))\n",
                "\n",
                "# Plot confusion matrix\n",
                "plt.figure(figsize=(12, 8))\n",
                "sns.heatmap(cm, annot=True, fmt=\"d\", cmap=\"Blues\", xticklabels=label_names, yticklabels=label_names)\n",
                "plt.xlabel(\"Predicted Label\")\n",
                "plt.ylabel(\"True Label\")\n",
                "plt.title(\"Confusion Matrix for ASL Sign Recognition\")\n",
                "plt.show()\n"
            ]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    f.write('\n')
print("Notebook updated successfully.")
