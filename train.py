
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models

# Project settings
DATASET_PATH = "dataset"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

print("Loading Currency Dataset...")

# Load training dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# Load validation dataset
val_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

class_names = train_data.class_names

print("Classes:", class_names)

if set(class_names) != {"fake", "genuine"}:
    raise ValueError("Create exactly two folders: fake and genuine")

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.cache().shuffle(500).prefetch(AUTOTUNE)
val_data = val_data.cache().prefetch(AUTOTUNE)

# CNN Model
model = models.Sequential([

    layers.Input(shape=(224, 224, 3)),

    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),

    layers.Rescaling(1./255),

    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.4),

    layers.Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("CNN Model Created Successfully!")

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Save trained model
model.save("currency_model.keras")

# Save class names
with open("classes.json", "w") as file:
    json.dump({"class_names": class_names}, file)

print("Model Training Completed!")
print("Model Saved Successfully!")
