import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, InputLayer, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Reshape
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from config import IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, CYRILLIC_LETTERS


def create_ffnn_model(input_shape, num_classes, activation='relu', optimizer='adam',
                      hidden_units=128, hidden_layers=2):
    """Create a feed-forward neural network model with configurable parameters"""
    print(f"\nBuilding FFNN with parameters:")
    print(f"- activation: {activation}")
    print(f"- optimizer: {optimizer}")
    print(f"- hidden_units: {hidden_units}")
    print(f"- hidden_layers: {hidden_layers}")

    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Flatten())

    # Hidden layers with configurable parameters
    for i in range(hidden_layers):
        # Use the specified hidden_units value for all layers
        units = hidden_units
        print(f"- Adding layer {i + 1} with {units} units")
        model.add(Dense(units, activation=activation))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    # Configure optimizer
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    elif optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Print model summary to verify configuration
    model.summary()

    return model


def create_tdnn_model(input_shape, num_classes, activation='relu', optimizer='adam',
                      hidden_units=128, hidden_layers=2):
    """Create a Time Delay Neural Network (TDNN) model with configurable parameters"""
    print(f"\nBuilding TDNN with parameters:")
    print(f"- activation: {activation}")
    print(f"- optimizer: {optimizer}")
    print(f"- hidden_units: {hidden_units}")
    print(f"- hidden_layers: {hidden_layers}")

    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))

    # Reshape the 2D image data into 1D sequential data
    # Treat each row of the image as a time step with IMG_WIDTH features
    model.add(Reshape((IMG_HEIGHT, IMG_WIDTH * IMG_CHANNELS)))

    # TDNN architecture - 1D convolutions with different dilation rates
    # to capture temporal patterns at different time scales
    kernel_sizes = [3, 5, 7]  # Different kernel sizes for different context windows
    filters_base = 16

    # First layer always has dilation rate of 1
    print(f"- Adding TDNN layer 1 with {filters_base} filters, kernel size {kernel_sizes[0]}, dilation rate 1")
    model.add(Conv1D(filters=filters_base,
                     kernel_size=kernel_sizes[0],
                     dilation_rate=1,
                     activation=activation,
                     padding='same'))
    model.add(MaxPooling1D(pool_size=2))

    # Additional TDNN layers with increasing dilation rates
    # Limited by hidden_layers parameter
    dilation_rate = 2
    filters = filters_base * 2

    for i in range(1, min(hidden_layers, 3)):  # Max 3 layers to avoid over-reduction
        kernel_idx = min(i, len(kernel_sizes) - 1)  # Ensure we don't go beyond kernel_sizes
        print(
            f"- Adding TDNN layer {i + 1} with {filters} filters, kernel size {kernel_sizes[kernel_idx]}, dilation rate {dilation_rate}")
        model.add(Conv1D(filters=filters,
                         kernel_size=kernel_sizes[kernel_idx],
                         dilation_rate=dilation_rate,
                         activation=activation,
                         padding='same'))
        model.add(MaxPooling1D(pool_size=2))

        # Increase parameters for next layer
        dilation_rate *= 2
        filters *= 2

    # Flatten features
    model.add(Flatten())

    # Fully connected layer using hidden_units parameter
    print(f"- Adding dense layer with {hidden_units} units")
    model.add(Dense(hidden_units, activation=activation))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    # Configure optimizer
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    elif optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    else:
        opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Print model summary to verify configuration
    model.summary()

    return model


def train_models(X_train, y_train, X_test, y_test, config, model_dir, train_generator=None, validation_generator=None):
    """Train both FFNN and TDNN models using generators if available"""
    # Print received configuration
    print(f"\nTraining with configuration: {config}")

    activation = config.get('activation', 'relu')
    optimizer = config.get('optimizer', 'adam')
    hidden_units = config.get('hidden_units', 128)
    hidden_layers = config.get('hidden_layers', 2)
    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 64)

    input_shape = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
    num_classes = len(CYRILLIC_LETTERS)

    print(f"Training models with {num_classes} output classes")
    print(f"Model parameters: activation={activation}, optimizer={optimizer}, "
          f"hidden_units={hidden_units}, hidden_layers={hidden_layers}, "
          f"epochs={epochs}, batch_size={batch_size}")

    # Create callbacks for improved training
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001, verbose=1),
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
    ]

    # FFNN model
    print("\n=== Training FFNN Model ===")
    ffnn_model = create_ffnn_model(
        input_shape=input_shape,
        num_classes=num_classes,
        activation=activation,
        optimizer=optimizer,
        hidden_units=hidden_units,
        hidden_layers=hidden_layers
    )

    # Add model checkpoint callback for FFNN
    ffnn_checkpoint = ModelCheckpoint(
        os.path.join(model_dir, 'ffnn_model_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    ffnn_callbacks = callbacks + [ffnn_checkpoint]

    # Train FFNN model
    if train_generator is not None and validation_generator is not None:
        print("Training FFNN with data generators...")
        ffnn_history = ffnn_model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=ffnn_callbacks,
            verbose=1
        )
    else:
        ffnn_history = ffnn_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=ffnn_callbacks,
            verbose=1
        )

    # TDNN model
    print("\n=== Training TDNN Model ===")
    tdnn_model = create_tdnn_model(
        input_shape=input_shape,
        num_classes=num_classes,
        activation=activation,
        optimizer=optimizer,
        hidden_units=hidden_units,
        hidden_layers=hidden_layers
    )

    # Add model checkpoint callback for TDNN
    tdnn_checkpoint = ModelCheckpoint(
        os.path.join(model_dir, 'tdnn_model_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    tdnn_callbacks = callbacks + [tdnn_checkpoint]

    # Train TDNN model
    if train_generator is not None and validation_generator is not None:
        print("Training TDNN with data generators...")
        tdnn_history = tdnn_model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=tdnn_callbacks,
            verbose=1
        )
    else:
        tdnn_history = tdnn_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=tdnn_callbacks,
            verbose=1
        )

    # Save both models
    ffnn_model_path = os.path.join(model_dir, 'ffnn_model')
    tdnn_model_path = os.path.join(model_dir, 'tdnn_model')

    # Use save method to save models
    ffnn_model.save(ffnn_model_path)
    tdnn_model.save(tdnn_model_path)

    # Also save in .keras format (more portable)
    ffnn_model.save(os.path.join(model_dir, 'ffnn_model.keras'))
    tdnn_model.save(os.path.join(model_dir, 'tdnn_model.keras'))

    print(f"FFNN model saved to {ffnn_model_path}")
    print(f"TDNN model saved to {tdnn_model_path}")

    return ffnn_model, tdnn_model, ffnn_history, tdnn_history


def load_models(model_dir):
    """Load pretrained FFNN and TDNN models with improved error handling"""
    ffnn_model = None
    tdnn_model = None

    # Check for models in different formats
    ffnn_model_path = os.path.join(model_dir, 'ffnn_model')
    tdnn_model_path = os.path.join(model_dir, 'tdnn_model')

    # Check for .keras format models
    ffnn_model_path_keras = os.path.join(model_dir, 'ffnn_model.keras')
    tdnn_model_path_keras = os.path.join(model_dir, 'tdnn_model.keras')

    # Check for best models (from checkpoints)
    ffnn_model_best = os.path.join(model_dir, 'ffnn_model_best.keras')
    tdnn_model_best = os.path.join(model_dir, 'tdnn_model_best.keras')

    # Check for .h5 format models (legacy support)
    ffnn_model_path_h5 = os.path.join(model_dir, 'ffnn_model.h5')
    tdnn_model_path_h5 = os.path.join(model_dir, 'tdnn_model.h5')

    # Try to load models in order of preference
    try:
        # First try to load best checkpoint models
        if os.path.exists(ffnn_model_best) and os.path.exists(tdnn_model_best):
            print("Loading from best checkpoint models...")
            ffnn_model = tf.keras.models.load_model(ffnn_model_best)
            tdnn_model = tf.keras.models.load_model(tdnn_model_best)
            print("Models successfully loaded from best checkpoints")

        # Then try SavedModel directories
        elif os.path.exists(ffnn_model_path) and os.path.exists(tdnn_model_path):
            print("Loading from SavedModel directories...")
            ffnn_model = tf.keras.models.load_model(ffnn_model_path)
            tdnn_model = tf.keras.models.load_model(tdnn_model_path)
            print("Models successfully loaded from SavedModel format")

        # Then try .keras format
        elif os.path.exists(ffnn_model_path_keras) and os.path.exists(tdnn_model_path_keras):
            print("Loading from .keras files...")
            ffnn_model = tf.keras.models.load_model(ffnn_model_path_keras)
            tdnn_model = tf.keras.models.load_model(tdnn_model_path_keras)
            print("Models successfully loaded from .keras format")

        # Finally try .h5 format
        elif os.path.exists(ffnn_model_path_h5) and os.path.exists(tdnn_model_path_h5):
            print("Loading from .h5 files...")
            ffnn_model = tf.keras.models.load_model(ffnn_model_path_h5)
            tdnn_model = tf.keras.models.load_model(tdnn_model_path_h5)
            print("Models successfully loaded from .h5 format")

        else:
            print("No models found in supported formats")
            return None, None

    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None

    return ffnn_model, tdnn_model
