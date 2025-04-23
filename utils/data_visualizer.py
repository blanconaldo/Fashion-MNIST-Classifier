import logging
import numpy as np
import matplotlib.pyplot as plt


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Visualization functions from data_visualization.py
def display_sample_images(images, labels, class_names, num_samples=5):
    """Display a few sample images with their class labels."""
    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()


def plot_class_distribution(y_train, class_names):
    """Plot the distribution of classes in the training set."""
    class_counts = np.bincount(y_train)
    plt.figure(figsize=(12, 6))
    plt.bar(class_names, class_counts)
    plt.title('Class Distribution in Training Set')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_pixel_distribution(x_train):
    """Plot the distribution of pixel intensities."""
    plt.figure(figsize=(10, 5))
    plt.hist(x_train.ravel(), bins=50, alpha=0.7)
    plt.title('Pixel Intensity Distribution')
    plt.xlabel('Pixel Value (Normalized)')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.show()


def plot_average_images(x_train, y_train, class_names):
    """Plot the average image for each class."""
    plt.figure(figsize=(15, 8))
    for i in range(10):
        class_indices = np.where(y_train == i)[0]
        class_images = x_train[class_indices]
        avg_image = np.mean(class_images, axis=0)

        plt.subplot(2, 5, i + 1)
        plt.imshow(avg_image, cmap='gray')
        plt.title(f'Avg {class_names[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_class_variations(x_train, y_train, class_names, selected_class=0):
    """Plot variations within a single class."""
    class_indices = np.where(y_train == selected_class)[0][:25]
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(class_indices):
        plt.subplot(5, 5, i + 1)
        plt.imshow(x_train[idx], cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Variations within {class_names[selected_class]} class', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


def visualize_dataset(x_train, y_train, x_val, y_val, class_names):
    """Run all visualizations using loaded data."""
    try:
        # Print dataset statistics
        logger.info("\n=== Dataset Statistics ===")
        logger.info(f"Training samples: {len(x_train)}")
        logger.info(f"Test samples: {len(x_val)}")
        logger.info(f"Test samples: {len(y_val)}")
        logger.info(f"Image dimensions: {x_train[0].shape}")
        logger.info(f"Number of classes: {len(class_names)}")

        logger.info("Displaying sample images...")
        display_sample_images(x_train, y_train, class_names)

        logger.info("Plotting class distribution...")
        plot_class_distribution(y_train, class_names)

        logger.info("Plotting pixel distribution...")
        plot_pixel_distribution(x_train)

        logger.info("Plotting average images per class...")
        plot_average_images(x_train, y_train, class_names)

        logger.info("Plotting variations within a class...")
        plot_class_variations(x_train, y_train, class_names)

        logger.info("All visualizations completed successfully.")
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        raise
