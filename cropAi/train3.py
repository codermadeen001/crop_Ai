import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Dataset path
DATA_DIR = r"C:\Users\inno\Documents\data"

# Simplified Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Data generators
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # Changed from 'sparse' to 'categorical'
    subset='training',
    shuffle=True,
    seed=123
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # Changed from 'sparse' to 'categorical'
    subset='validation',
    shuffle=False,
    seed=123
)

# Get class information
class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)
print("Classes detected:", class_names)

# Calculate class weights
class_counts = np.bincount(train_generator.classes)
class_weights = {i: 1./count for i, count in enumerate(class_counts)}
total = sum(class_weights.values())
class_weights = {i: (weight/total)*num_classes for i, weight in class_weights.items()}

# Model Architecture
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Learning rate
optimizer = Adam(learning_rate=0.0001)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',  # Changed from 'sparse_categorical_crossentropy'
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
]

# Training
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=30,
    callbacks=callbacks,
    class_weight=class_weights
)

# Save model
model.save("maize_disease_detector.h5")
print("✅ Model saved successfully")

# Evaluation
print("\nEvaluation results:")
results = model.evaluate(val_generator)
print(f"Validation Accuracy: {results[1]*100:.2f}%")