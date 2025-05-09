import os
import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm import tqdm
import re

# Import project modules
from config import CYRILLIC_LETTERS, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, MODELS_PATH
from utils import convert_drawn_image, load_model_config

# Path configurations
PROJECT_ROOT = r"C:\Neiro\NeiroCirill1.1"
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "CyrillicTest")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "AllTest")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_test_images():
    """Load test images from CyrillicTest directory in random order"""
    print(f"Loading test images from {TEST_DATA_PATH}")

    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Test data directory {TEST_DATA_PATH} does not exist")
        sys.exit(1)

    # Check if the folder structure matches expectations
    if not any(os.path.isdir(os.path.join(TEST_DATA_PATH, letter)) for letter in CYRILLIC_LETTERS):
        print(f"Error: Test directory structure doesn't match expected Cyrillic letters")
        print(f"Expected folders named: {', '.join(CYRILLIC_LETTERS)}")
        sys.exit(1)

    # Prepare a list to store test image paths and corresponding labels
    test_data = []

    # Loop through each letter folder
    for letter in CYRILLIC_LETTERS:
        letter_dir = os.path.join(TEST_DATA_PATH, letter)

        if not os.path.exists(letter_dir):
            print(f"Warning: Directory for letter '{letter}' not found in test data")
            continue

        # Get all png files in this letter's directory
        png_files = [f for f in os.listdir(letter_dir) if f.lower().endswith('.png')]

        if not png_files:
            print(f"Warning: No PNG images found for letter '{letter}'")
            continue

        # Add each image path and its true label (letter)
        for png_file in png_files:
            test_data.append({
                'path': os.path.join(letter_dir, png_file),
                'true_label': letter,
                'true_index': CYRILLIC_LETTERS.index(letter)
            })

    # Shuffle the test data
    random.shuffle(test_data)

    print(
        f"Loaded {len(test_data)} test images from {len(set(item['true_label'] for item in test_data))} letter classes")
    return test_data


def load_all_models():
    """Load all models from the Models directory"""
    print(f"Loading models from {MODELS_PATH}")

    if not os.path.exists(MODELS_PATH):
        print(f"Error: Models directory {MODELS_PATH} does not exist")
        sys.exit(1)

    models_info = []
    model_dirs = [d for d in os.listdir(MODELS_PATH) if os.path.isdir(os.path.join(MODELS_PATH, d))]

    if not model_dirs:
        print(f"Error: No model directories found in {MODELS_PATH}")
        sys.exit(1)

    print(f"Found {len(model_dirs)} model directories")

    for model_dir_name in model_dirs:
        model_dir = os.path.join(MODELS_PATH, model_dir_name)

        # Try to load FFNN model
        ffnn_path = os.path.join(model_dir, 'ffnn_model.keras')
        if os.path.exists(ffnn_path):
            try:
                ffnn_model = tf.keras.models.load_model(ffnn_path)
                # Load model configuration if available
                config = load_model_config(model_dir) or {}

                # Extract model parameters for more descriptive name
                params = []
                if config:
                    if 'activation' in config:
                        params.append(f"act={config['activation']}")
                    if 'optimizer' in config:
                        params.append(f"opt={config['optimizer']}")
                    if 'hidden_units' in config:
                        params.append(f"units={config['hidden_units']}")
                    if 'hidden_layers' in config:
                        params.append(f"layers={config['hidden_layers']}")

                param_str = ", ".join(params) if params else "default_params"
                display_name = f"FFNN: {model_dir_name} ({param_str})"

                models_info.append({
                    'name': display_name,
                    'model': ffnn_model,
                    'type': 'FFNN',
                    'dir_name': model_dir_name,
                    'config': config
                })
                print(f"Loaded FFNN model from {model_dir_name}")
            except Exception as e:
                print(f"Error loading FFNN model from {model_dir_name}: {str(e)}")

        # Try to load TDNN model
        tdnn_path = os.path.join(model_dir, 'tdnn_model.keras')
        if os.path.exists(tdnn_path):
            try:
                tdnn_model = tf.keras.models.load_model(tdnn_path)
                # Load model configuration if available
                config = load_model_config(model_dir) or {}

                # Extract model parameters for more descriptive name
                params = []
                if config:
                    if 'activation' in config:
                        params.append(f"act={config['activation']}")
                    if 'optimizer' in config:
                        params.append(f"opt={config['optimizer']}")
                    if 'hidden_units' in config:
                        params.append(f"units={config['hidden_units']}")
                    if 'hidden_layers' in config:
                        params.append(f"layers={config['hidden_layers']}")

                param_str = ", ".join(params) if params else "default_params"
                display_name = f"TDNN: {model_dir_name} ({param_str})"

                models_info.append({
                    'name': display_name,
                    'model': tdnn_model,
                    'type': 'TDNN',
                    'dir_name': model_dir_name,
                    'config': config
                })
                print(f"Loaded TDNN model from {model_dir_name}")
            except Exception as e:
                print(f"Error loading TDNN model from {model_dir_name}: {str(e)}")

    if not models_info:
        print("Error: No models could be loaded")
        sys.exit(1)

    print(f"Successfully loaded {len(models_info)} models")
    return models_info


def preprocess_image(image_path):
    """Preprocess a test image to the format expected by the models"""
    try:
        # Load image using PIL
        img = Image.open(image_path).convert('L')  # Convert to grayscale

        # Resize to target dimensions
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)

        # Convert to array and normalize
        img_array = np.array(img)
        img_array = img_array / 255.0

        # Reshape to model input format
        return img_array.reshape(1, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {str(e)}")
        return None


def evaluate_models(models_info, test_data):
    """Evaluate all models on the test dataset"""
    results = []

    print("Evaluating models on test data...")

    for model_info in models_info:
        model = model_info['model']
        model_name = model_info['name']

        print(f"\nEvaluating {model_name}...")

        correct_predictions = 0
        total_predictions = 0

        # Keep detailed predictions for analysis
        letter_stats = {letter: {'correct': 0, 'total': 0} for letter in CYRILLIC_LETTERS}

        # Process test images with a progress bar
        for test_item in tqdm(test_data, desc=f"Testing {model_info['type']}"):
            image_path = test_item['path']
            true_label = test_item['true_label']
            true_index = test_item['true_index']

            # Preprocess the image
            input_img = preprocess_image(image_path)
            if input_img is None:
                continue

            # Make prediction
            try:
                predictions = model.predict(input_img, verbose=0)[0]
                pred_index = np.argmax(predictions)
                pred_label = CYRILLIC_LETTERS[pred_index] if pred_index < len(CYRILLIC_LETTERS) else "Unknown"

                is_correct = (pred_label == true_label)
                if is_correct:
                    correct_predictions += 1

                total_predictions += 1

                # Update letter-specific statistics
                letter_stats[true_label]['total'] += 1
                if is_correct:
                    letter_stats[true_label]['correct'] += 1

            except Exception as e:
                print(f"Error predicting image {image_path}: {str(e)}")

        # Calculate overall accuracy
        if total_predictions > 0:
            accuracy = (correct_predictions / total_predictions) * 100
        else:
            accuracy = 0

        # Calculate per-letter accuracy
        letter_accuracy = {}
        for letter, stats in letter_stats.items():
            if stats['total'] > 0:
                letter_accuracy[letter] = (stats['correct'] / stats['total']) * 100
            else:
                letter_accuracy[letter] = 0

        # Create result entry
        result = {
            'model_name': model_name,
            'type': model_info['type'],
            'dir_name': model_info['dir_name'],
            'accuracy': accuracy,
            'correct': correct_predictions,
            'total': total_predictions,
            'letter_accuracy': letter_accuracy,
            'config': model_info['config']
        }

        results.append(result)

        print(f"{model_name}: Accuracy = {accuracy:.2f}% ({correct_predictions}/{total_predictions})")

    return results


def create_comparison_chart(results):
    """Create bar chart comparing model accuracies"""
    if not results:
        print("No results to plot")
        return

    # Sort results by accuracy (highest first)
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)

    # Prepare data for plotting
    model_names = [r['model_name'] for r in sorted_results]
    accuracies = [r['accuracy'] for r in sorted_results]
    model_types = [r['type'] for r in sorted_results]

    # Define colors based on model type
    colors = ['#3498db' if t == 'FFNN' else '#e74c3c' for t in model_types]

    # Create figure (with increased size for better readability)
    plt.figure(figsize=(14, 8))

    # Create bar chart
    bars = plt.bar(range(len(model_names)), accuracies, color=colors)

    # Add percentage labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{accuracies[i]:.2f}%',
                 ha='center', va='bottom', rotation=0)

    # Add labels and title
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.title('Comparison of Model Accuracies on Test Dataset')

    # Create a custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='FFNN Models'),
        Patch(facecolor='#e74c3c', label='TDNN Models')
    ]
    plt.legend(handles=legend_elements)

    # Set x-axis labels with appropriate rotation for readability
    plt.xticks(range(len(model_names)), [re.sub(r'\(.*?\)', '', name).strip() for name in model_names], rotation=45,
               ha='right')

    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout to make room for labels
    plt.tight_layout()

    # Save the chart
    output_path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
    plt.savefig(output_path, dpi=300)
    print(f"Chart saved to {output_path}")

    # Save detailed results to CSV
    df = pd.DataFrame([{
        'Model': r['model_name'],
        'Type': r['type'],
        'Directory': r['dir_name'],
        'Accuracy': r['accuracy'],
        'Correct': r['correct'],
        'Total': r['total']
    } for r in results])

    csv_path = os.path.join(OUTPUT_DIR, 'model_comparison_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Detailed results saved to {csv_path}")

    # Save per-letter accuracy details
    letter_data = []
    for r in results:
        model_data = {'Model': r['model_name']}
        for letter, accuracy in r['letter_accuracy'].items():
            model_data[letter] = accuracy
        letter_data.append(model_data)

    letter_df = pd.DataFrame(letter_data)
    letter_csv_path = os.path.join(OUTPUT_DIR, 'per_letter_accuracy.csv')
    letter_df.to_csv(letter_csv_path, index=False)
    print(f"Per-letter accuracy saved to {letter_csv_path}")


def main():
    print("Cyrillic Neural Network Model Comparison Tool")
    print("=" * 50)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load test data
    test_data = load_test_images()

    # Load all models
    models_info = load_all_models()

    # Evaluate models
    results = evaluate_models(models_info, test_data)

    # Create comparison chart
    create_comparison_chart(results)

    print("\nModel comparison complete! Results are available in:")
    print(f"- Chart: {os.path.join(OUTPUT_DIR, 'model_comparison.png')}")
    print(f"- Overall results: {os.path.join(OUTPUT_DIR, 'model_comparison_results.csv')}")
    print(f"- Per-letter accuracy: {os.path.join(OUTPUT_DIR, 'per_letter_accuracy.csv')}")


if __name__ == "__main__":
    main()