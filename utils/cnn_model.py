import os
import logging
import keras
from keras import callbacks, models, layers
from data_augmentation import create_data_augmentation_generator

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_data_for_cnn(x_train, x_val, x_test):
    # Add channel dimension if it doesn't exist
    if len(x_train.shape) == 3:
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    logger.info(f"Data reshaped for CNN input: {x_train.shape}")
    return x_train, x_val, x_test


def build_cnn_model():
    model = models.Sequential([
        # Input Layer - images remain in 2D structure (28x28x1)
        layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second block
        layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Dense layers
        layers.Flatten(),
        layers.Dense(512, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer='adam',  # Adaptive learning rate
        loss=keras.losses.sparse_categorical_crossentropy,  # For integer labels
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    # Print model summary
    model.summary()

    return model


def train_model(model, x_train, y_train, x_val, y_val, batch_size=64, epochs=80, augment=True):
    # Create a model checkpoint to save best weights
    checkpoint_path = "../models/fashion_mnist_model_best.keras"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Define callbacks
    callbacks_list = [
        # Save best model based on validation accuracy
        callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_sparse_categorical_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Stop training when no improvement
        callbacks.EarlyStopping(
            monitor='val_sparse_categorical_accuracy',
            patience=5,
            mode='max',
            verbose=1
        ),
        # Reduce learning rate when plateau reached
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=3,
            mode='min',
            verbose=1
        )
    ]

    # Use data augmentation if specified
    if augment:
        logger.info("Training with data augmentation...")
        datagen = create_data_augmentation_generator()
        datagen.fit(x_train)

        # Train with augmentation
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks_list,
            steps_per_epoch=len(x_train) // batch_size
        )
    else:
        logger.info("Training without data augmentation...")
        # Train without augmentation
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks_list,
            shuffle=True
        )

    # Load best weights
    model.load_weights(checkpoint_path)
    logger.info("Model training completed. Best weights loaded.")

    return model, history
