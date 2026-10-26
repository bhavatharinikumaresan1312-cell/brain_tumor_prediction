import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf  # type: ignore[import]
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore[import]
from tensorflow.keras.applications import EfficientNetB0  # type: ignore[import]
from tensorflow.keras.applications.efficientnet import preprocess_input  # type: ignore[import]
from tensorflow.keras.models import Model  # type: ignore[import]
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D  # type: ignore[import]
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore[import]
from sklearn.metrics import confusion_matrix, classification_report  # type: ignore[import]
from sklearn.utils.class_weight import compute_class_weight  # type: ignore[import]

# -------------------------------
# Paths & Constants
# -------------------------------
train_dir = "brain_tumor_dataset/Training"
test_dir = "brain_tumor_dataset/Testing"

IMG_SIZE = (150, 150)    # smaller image -> faster
BATCH_SIZE = 16
# Allow a quick smoke test by setting environment variable SMOKE_TEST=1
SMOKE_TEST = os.getenv("SMOKE_TEST", "0") == "1"
EPOCHS = 1 if SMOKE_TEST else 8  # fewer epochs for speed (1 when smoke testing)
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# -------------------------------
# Data Generators (use model-specific preprocessing)
# -------------------------------
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
test_data = test_gen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

print("\nClass mapping:", train_data.class_indices)

# -------------------------------
# Build Model (EfficientNetB0)
# -------------------------------
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))

for layer in base_model.layers[:-20]:   # keep last few trainable
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(len(class_labels), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# Training setup: callbacks, class weights and dynamic steps
# -------------------------------
# compute class weights to handle imbalance
class_indices = train_data.classes
classes = np.unique(class_indices)
class_weights_raw = compute_class_weight(class_weight='balanced', classes=classes, y=class_indices)
class_weights = {int(c): float(w) for c, w in zip(classes, class_weights_raw)}

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
ckpt = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

if SMOKE_TEST:
    steps_per_epoch = min(len(train_data), 5)
    validation_steps = min(len(test_data), 2)
else:
    steps_per_epoch = min(len(train_data), 100)
    validation_steps = min(len(test_data), 20)

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    class_weight=class_weights,
    callbacks=[early_stop, ckpt]
)

# -------------------------------
# Save Accuracy Graph
# -------------------------------
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("training_accuracy_plot_fast.png")
plt.close()

# -------------------------------
# Evaluate & Save
# -------------------------------
y_true = test_data.classes
y_pred = np.argmax(model.predict(test_data), axis=1)
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_labels))

model.save("brain_tumor_multiclass_model.h5")
print("✅ Fast model trained and saved successfully!")

# -------------------------------
# Optional: Test Single Image
# -------------------------------
def predict_image(img_path, model_instance=model, labels=class_labels):
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found:", img_path)
        return
    img = cv2.resize(img, IMG_SIZE)
    img = np.expand_dims(img, axis=0) / 255.0
    preds = model_instance.predict(img)
    label_index = np.argmax(preds)
    confidence = float(np.max(preds))
    print(f"🧠 Prediction: {labels[label_index]} ({confidence*100:.2f}%)")

example_img = "brain_tumor_dataset/Testing/glioma/Te-gl_0010.jpg"
predict_image(example_img)
from PIL import Image, ImageOps
import streamlit as st  