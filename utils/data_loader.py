import os
import pickle
import logging
from sklearn.model_selection import train_test_split

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_fashion_mnist(cache_dir='./data'):
    # Define class names for the dataset
    label_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'fashion_mnist_data.pkl')

    try:
        # Check if the dataset is already cached
        if os.path.exists(cache_file):
            logger.info("Loading Fashion MNIST from cache...")
            with open(cache_file, 'rb') as f_in:
                x_train, y_train, x_test, y_test = pickle.load(f_in)
            logger.info("Dataset loaded from cache successfully.")
        else:
            # Load the dataset from TensorFlow with better error handling
            logger.info("Loading Fashion MNIST from TensorFlow...")
            try:
                # Try importing directly from tensorflow
                import keras
                (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
            except (ImportError, AttributeError) as e:
                logger.warning(f"Error with TensorFlow import: {e}")
                try:
                    # Try importing keras directly as fallback
                    import keras
                    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
                except (ImportError, AttributeError):
                    error_msg = "Could not import keras. Please install TensorFlow or Keras."
                    logger.error(error_msg)
                    raise ImportError(error_msg)

            # Normalize the data to the range [0, 1]
            x_train = x_train.astype('float32') / 255.0
            x_test = x_test.astype('float32') / 255.0

            # Cache the dataset for future use
            try:
                with open(cache_file, 'wb') as f_out:
                    pickle.dump((x_train, y_train, x_test, y_test), f_out)
                logger.info("Dataset cached successfully.")
            except Exception as e:
                logger.warning(f"Could not cache dataset: {e}")

        logger.info(f"Training data shape: {x_train.shape}, Labels shape: {y_train.shape}")
        logger.info(f"Testing data shape: {x_test.shape}, Labels shape: {y_test.shape}")

        # Shuffle and split training data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42, shuffle=True)
        logger.info(f"After splitting - Training: {x_train.shape}, Validation: {x_val.shape}")

        return x_train, y_train, x_val, y_val, x_test, y_test, label_names

    except Exception as e:
        logger.error(f"Error loading Fashion MNIST dataset: {e}")
        raise
    finally:
        logger.info("Fashion MNIST dataset loading process completed.")
