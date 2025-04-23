import sys
from utils.data_loader import *
from utils.data_visualizer import *
from utils.data_augmentation import *
from utils.cnn_model import *
from utils.model_evaluation import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    try:
        # Load the Fashion MNIST dataset
        logger.info("Starting Fashion MNIST data loading...")
        x_train, y_train, x_val, y_val, x_test, y_test, class_names = load_fashion_mnist()

        # Visualize the dataset
        logger.info("\n=== Visualizing Original Dataset ===")
        visualize_dataset(x_train, y_train, x_val, y_val, class_names)

        # 3. Prepare data for CNN (reshape to include channel dimension)
        logger.info("=== Preparing Data for CNN ===")
        x_train, x_val, x_test = prepare_data_for_cnn(x_train, x_val, x_test)
        logger.info(f"Training data shape: {x_train.shape}")
        logger.info(f"Test data shape: {x_test.shape}")

        # Demonstrate data augmentation
        logger.info("\n=== Demonstrating Data Augmentation ===")
        visualize_augmented_samples(x_train, y_train, class_names, num_samples=5)

        # Building CNN model
        logger.info("=== Building CNN Model ===")
        model = build_cnn_model()

        # Check time before training
        start_time = time.time()

        # Training the model
        logger.info("=== Training CNN Model ===")
        model, history = train_model(
            model,
            x_train, y_train,
            x_val, y_val,
            batch_size=64,
            epochs=80,
            augment=True
        )

        # After training completes
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

        # Evaluating the model
        logger.info("=== Evaluating CNN Model ===")
        evaluate_model(model, x_test, y_test, class_names, history=history)

        logger.info("Fashion MNIST CNN project completed successfully!")

        # Saving entire model
        model_path, keras_path = save_complete_model(
            model,
            model_name="fashion_mnist_cnn"
        )
        logger.info(f"Model saved for deployment at: {model_path}")
        logger.info(f"Model saved in Keras format at: {keras_path}")



    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
