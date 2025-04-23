import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, x_test, y_test, class_names, history=None):
    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    logger.info(f"Test accuracy: {test_acc:.4f}")

    # Generate predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # Classification report
    report = classification_report(y_test, y_pred_classes,
                                   target_names=class_names)
    logger.info(f"Classification Report:\n{report}")

    # Plot training history (if available)
    if history is not None and hasattr(history, 'history'):
        # Plot accuracy
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['sparse_categorical_accuracy'])
        plt.plot(history.history['val_sparse_categorical_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.tight_layout()
        plt.show()

    # Show examples of predictions
    visualize_predictions(model, x_test, y_test, class_names)


def visualize_predictions(model, x_test, y_test, class_names, num_images=10):
    # Get random images from test set
    indices = np.random.choice(len(x_test), num_images, replace=False)
    test_images = x_test[indices]
    test_labels = y_test[indices]

    # Get predictions
    predictions = model.predict(test_images)
    pred_labels = np.argmax(predictions, axis=1)

    # Plot images with predictions
    plt.figure(figsize=(15, 3 * (num_images // 5 + 1)))
    for i, idx in enumerate(range(num_images)):
        plt.subplot(num_images // 5 + 1, 5, i + 1)
        img = test_images[idx].reshape(28, 28)  # Reshape for visualization
        plt.imshow(img, cmap='gray')

        # Green text for correct predictions, red for incorrect
        color = 'green' if pred_labels[idx] == test_labels[idx] else 'red'
        pred_class = class_names[pred_labels[idx]]
        true_class = class_names[test_labels[idx]]

        plt.title(f"Pred: {pred_class}\nTrue: {true_class}", color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def save_complete_model(model, base_path="./models", model_name="fashion_mnist_model"):

    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)

    # Add timestamp for versioning
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 1. Save complete model in .keras format (new recommended format)
    keras_path = f"{base_path}/{model_name}_{timestamp}.keras"
    try:
        model.save(keras_path)
        logger.info(f"Complete model saved in Keras format at: {keras_path}")
    except Exception as e:
        logger.error(f"Error saving model in Keras format: {e}")
        keras_path = None

    # 2. Also save just the weights (useful for transfer learning)
    weights_path = f"{base_path}/{model_name}_{timestamp}.weights.h5"
    try:
        model.save_weights(weights_path)
        logger.info(f"Model weights saved at: {weights_path}")
    except Exception as e:
        logger.error(f"Error saving model weights: {e}")
        weights_path = None

    # 3. Save a 'latest' version for easy reference
    latest_path = f"{base_path}/{model_name}_latest.keras"
    try:
        model.save(latest_path)
        logger.info(f"Latest model saved at: {latest_path}")
    except Exception as e:
        logger.error(f"Error saving latest model: {e}")

    return keras_path, weights_path