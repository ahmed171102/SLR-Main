import json

file_path = r'm:\Term 10\Grad\SLR Main\Letters\Merger Notebook new approach\Trials of the model\mobile-net-v1-2.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 25
cell_25_source = '''from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Data generators
# We use preprocess_input for MobileNetV2 instead of rescale=1./255
# We also separate validation datagen so it doesn't get augmented!
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False, # Horizontal flip is bad for ASL (changes hand direction)
    validation_split=0.2  # Splitting data into train (80%) and val (20%)
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

# Train & validation generators (load images directly from disk)
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    seed=42
)

print("Class labels:", train_generator.class_indices)'''

# Update Cell 31
cell_31_source = '''for layer in base_model.layers[-20:]:  # Unfreeze last 20 layers
    # DO NOT unfreeze BatchNormalization layers! It destroys their moving averages!
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

# Recompile with a lower learning rate to avoid overfitting
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_LEARN_RATE),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=FINETUNE_EPOCHS
)'''

# Update Cell 37
cell_37_source = '''from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Point to the newly created folder!
test_folder = r"/kaggle/working/structured_test"

# Use preprocess_input instead of rescale=1./255
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    test_folder,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False, # Very important for evaluation
    classes=list(train_generator.class_indices.keys()) # Forces exact class alignment with training!
)

print(f"Loaded test images from {test_folder}")'''

def find_and_replace_cell(nb, snippet, new_source):
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if snippet in source:
                cell['source'] = [line + '\n' for line in new_source.split('\n')]
                if cell['source'] and cell['source'][-1].endswith('\n'):
                    cell['source'][-1] = cell['source'][-1][:-1]
                return True
    return False

# Attempt replacements
find_and_replace_cell(nb, 'train_datagen = ImageDataGenerator(\n    rescale=1./255,  # Normalize', cell_25_source)
find_and_replace_cell(nb, 'for layer in base_model.layers[-20:]:  # Unfreeze last 20 layers\n    layer.trainable = True', cell_31_source)
find_and_replace_cell(nb, 'test_datagen = ImageDataGenerator(rescale=1./255)', cell_37_source)

# Cell 47 targeted replacement
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'hand_resized = np.expand_dims(hand_resized, axis=0) / 255.0  # Normalize' in source:
            old_snippet = '''            if hand_crop.shape[0] > 0 and hand_crop.shape[1] > 0:
                hand_resized = cv2.resize(hand_crop, (128, 128))
                hand_resized = np.expand_dims(hand_resized, axis=0) / 255.0  # Normalize'''
            new_snippet = '''            if hand_crop.shape[0] > 0 and hand_crop.shape[1] > 0:
                from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
                hand_resized = cv2.resize(hand_crop, (128, 128))
                hand_resized = np.expand_dims(hand_resized, axis=0).astype('float32')
                hand_resized = preprocess_input(hand_resized)  # Correct MobileNetV2 normalization'''
            
            new_source = source.replace(old_snippet, new_snippet)
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            if cell['source'] and cell['source'][-1].endswith('\n'):
                cell['source'][-1] = cell['source'][-1][:-1]

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Notebook successfully updated in the M: drive!')
