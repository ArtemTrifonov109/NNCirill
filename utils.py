import os
import time
import json
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import CYRILLIC_PATH, CYRILLIC_LETTERS, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS


def load_dataset():
    """Load and preprocess the Cyrillic dataset using ImageDataGenerator"""
    print(f"Loading dataset from {CYRILLIC_PATH}")

    # Check if the base directory exists
    if not os.path.exists(CYRILLIC_PATH):
        print(f"Warning: Directory {CYRILLIC_PATH} does not exist. Creating it...")
        os.makedirs(CYRILLIC_PATH, exist_ok=True)
        return np.array([]), np.array([])

    # List actual folders in directory to verify order
    actual_folders = sorted([d for d in os.listdir(CYRILLIC_PATH)
                             if os.path.isdir(os.path.join(CYRILLIC_PATH, d))])
    print("\nActual folder order in filesystem:")
    print(actual_folders)
    print("\nExpected order from CYRILLIC_LETTERS:")
    print(CYRILLIC_LETTERS)

    if actual_folders != CYRILLIC_LETTERS:
        print("\nWARNING: Folder order doesn't match CYRILLIC_LETTERS order!")
        print("This may cause incorrect letter predictions.")

    # Count samples for each class
    # ... keep existing code (class counting)

    # Create train data generator
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        CYRILLIC_PATH,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        color_mode='grayscale',
        batch_size=32,
        class_mode='sparse',
        subset='training',
        shuffle=True
    )

    # Load validation data
    validation_generator = train_datagen.flow_from_directory(
        CYRILLIC_PATH,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        color_mode='grayscale',
        batch_size=32,
        class_mode='sparse',
        subset='validation',
        shuffle=True
    )

    # Print detailed class mapping
    print("\nDetailed class mapping:")
    for letter, idx in train_generator.class_indices.items():
        print(f"Folder '{letter}' -> Class index {idx}")

    # Verify label range
    print("Verifying label range...")
    for i in range(min(5, len(train_generator))):
        batch_x, batch_y = next(train_generator)
        print(f"Batch {i} labels: min={np.min(batch_y)}, max={np.max(batch_y)}")
        if np.max(batch_y) >= len(CYRILLIC_LETTERS):
            print(f"WARNING: Found label {np.max(batch_y)} which is >= {len(CYRILLIC_LETTERS)}")

    # Get a batch of samples to return for immediate training
    X_batch, y_batch = next(train_generator)
    for i in range(min(10, len(train_generator) - 1)):  # Get more batches to fill the array
        X_temp, y_temp = next(train_generator)
        X_batch = np.vstack((X_batch, X_temp))
        y_batch = np.append(y_batch, y_temp)

    # Get validation samples
    X_val, y_val = next(validation_generator)
    for i in range(min(5, len(validation_generator) - 1)):  # Get more batches
        X_temp, y_temp = next(validation_generator)
        X_val = np.vstack((X_val, X_temp))
        y_val = np.append(y_val, y_temp)

    print(f"Loaded {X_batch.shape[0]} training samples and {X_val.shape[0]} validation samples")
    return X_batch, y_batch, X_val, y_val, train_generator, validation_generator


def convert_drawn_image(image, size=(IMG_WIDTH, IMG_HEIGHT)):
    """Convert drawn image to model input format with minimal preprocessing"""
    try:
        # Convert to grayscale
        img = image.convert('L')

        # Resize to target dimensions
        img = img.resize(size, Image.Resampling.LANCZOS)

        # Normalize values
        img_array = np.array(img)
        img_array = img_array / 255.0

        return img_array.reshape(1, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    except Exception as e:
        print(f"Error converting drawn image: {str(e)}")
        # Return empty image in case of error
        blank = np.zeros((1, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
        return blank


def save_model_config(model_dir, config):
    """Save model configuration to file"""
    print(f"Saving config to {model_dir}: {config}")
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def load_model_config(model_dir):
    """Load model configuration from file"""
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            print(f"Loaded config from {model_dir}: {config}")
            return config
    return None


def get_letter_prediction(model, image, top_k=5):
    """Get letter prediction from model"""
    try:
        predictions = model.predict(image, verbose=0)[0]
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        results = []

        for idx in top_indices:
            if idx < len(CYRILLIC_LETTERS):
                letter = CYRILLIC_LETTERS[idx]
                confidence = predictions[idx] * 100
                results.append((letter, confidence))

        return results
    except Exception as e:
        print(f"Error getting letter prediction: {str(e)}")
        return []


def plot_training_history(history, model_dir):
    """Plot and save training history"""
    try:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')

        plt.tight_layout()

        # Create subdirectory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        output_file = os.path.join(model_dir, 'training_history.png')
        plt.savefig(output_file)
        plt.close()
        print(f"Training history plot saved to {output_file}")
    except Exception as e:
        print(f"Error plotting training history: {str(e)}")