import logging
import numpy as np
import matplotlib.pyplot as plt
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data Augmentation Functions
def create_data_augmentation_generator(rotation_range=10,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       zoom_range=0.1,
                                       horizontal_flip=True,
                                       shear_range=0.1):

    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        shear_range=shear_range,
        fill_mode='nearest'
    )

    logger.info(f"Created data augmentation generator with parameters:")
    logger.info(f"  - Rotation range: {rotation_range} degrees")
    logger.info(f"  - Width shift range: {width_shift_range}")
    logger.info(f"  - Height shift range: {height_shift_range}")
    logger.info(f"  - Zoom range: {zoom_range}")
    logger.info(f"  - Horizontal flip: {horizontal_flip}")
    logger.info(f"  - Shear range: {shear_range}")

    return datagen


def apply_augmentation(x_train, y_train, batch_size=32, seed=42):
    # Reshape the data for the generator if needed (add channel dimension)
    if len(x_train.shape) == 3:
        x_train_reshaped = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    else:
        x_train_reshaped = x_train

    # Create augmentation generator
    datagen = create_data_augmentation_generator()

    # Fit the generator on the data (compute statistics for normalization)
    datagen.fit(x_train_reshaped)

    # Return the generator
    return datagen.flow(x_train_reshaped, y_train, batch_size=batch_size, seed=seed)


def visualize_augmented_samples(x_train, y_train, class_names, num_samples=5):
    try:
        # Reshape data for the generator if needed
        if len(x_train.shape) == 3:
            x_train_reshaped = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        else:
            x_train_reshaped = x_train

        # Select a random class
        selected_class = np.random.randint(0, len(class_names))
        class_indices = np.where(y_train == selected_class)[0]

        # Get a sample image from the selected class
        sample_idx = np.random.choice(class_indices)
        sample_image = x_train_reshaped[sample_idx:sample_idx + 1]

        # Create augmentation generator
        datagen = create_data_augmentation_generator()

        # Create an iterator that generates augmented versions of the sample image
        augmented_iterator = datagen.flow(sample_image, batch_size=1, seed=42)

        # Plot original and augmented images
        plt.figure(figsize=(12, 4))

        # Plot original
        plt.subplot(1, num_samples + 1, 1)
        plt.imshow(x_train[sample_idx].reshape(28, 28), cmap='gray')
        plt.title(f"Original\n{class_names[selected_class]}")
        plt.axis("off")

        # Plot augmented versions
        for i in range(num_samples):
            augmented = next(augmented_iterator)[0]
            plt.subplot(1, num_samples + 1, i + 2)
            plt.imshow(augmented.reshape(28, 28), cmap='gray')
            plt.title(f"Augmented {i + 1}")
            plt.axis("off")

        plt.suptitle(f"Data Augmentation Examples for {class_names[selected_class]}")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.error(f"Error visualizing augmented samples: {e}")
        raise
