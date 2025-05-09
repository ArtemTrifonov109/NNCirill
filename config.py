import os

# Project paths
PROJECT_ROOT = r"C:\Neiro\NeiroCirill1.1"
CYRILLIC_PATH = os.path.join(PROJECT_ROOT, "Cyrillic")
MODELS_PATH = os.path.join(PROJECT_ROOT, "Models")
STATISTICS_PATH = os.path.join(PROJECT_ROOT, "Statistics")

# Ensure directories exist
for path in [PROJECT_ROOT, CYRILLIC_PATH, MODELS_PATH, STATISTICS_PATH]:
    os.makedirs(path, exist_ok=True)

# Model configuration options
ACTIVATION_FUNCTIONS = ["relu", "tanh", "sigmoid", "elu"]
OPTIMIZERS = ["adam", "rmsprop", "sgd"]
HIDDEN_UNITS_OPTIONS = [32, 64, 128, 256, 512, 1024, 2048]  # Updated with more options from 32 to 2048
HIDDEN_LAYERS_OPTIONS = [1, 2, 3, 4, 5]  # Updated to include options from 1 to 5

IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_CHANNELS = 1

CYRILLIC_LETTERS = [
    'Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й',
    'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У',
    'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э',
    'Ю', 'Я'
]

# Default model parameters
DEFAULT_PARAMETERS = {
    "activation": "relu",
    "optimizer": "adam",
    "hidden_units": 256,
    "hidden_layers": 2,
    "epochs": 50,
    "batch_size": 32
}