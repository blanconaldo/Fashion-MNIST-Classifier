import sys
import numpy as np
from keras.src.saving import load_model
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, messagebox
from PIL import Image, ImageTk
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Class names from Fashion MNIST
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def load_fashion_model(model_path="./models/fashion_mnist_model_best.keras"):
    try:
        logger.info(f"Loading model from {model_path}")
        model = load_model(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def preprocess_image(image_path):
    # Preprocess the uploaded image to match Fashion MNIST format.
    try:
        # Open the image
        img = Image.open(image_path)

        # Convert to grayscale
        img = img.convert('L')

        # Resize to 28x28
        img = img.resize((28, 28))

        # Convert to numpy array and normalize
        img_array = np.array(img)
        img_array = img_array / 255.0

        # Invert colors if needed (Fashion MNIST has white items on black background)
        if np.mean(img_array) > 0.5:
            img_array = 1 - img_array

        # Expand dimensions to match model input shape (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)

        return img_array, img
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None, None

class FashionClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fashion MNIST Classifier")
        self.root.geometry("600x550")
        self.root.configure(bg="#f0f0f0")

        # Initialize instance variables
        self.img_frame = None
        self.display_label = None
        self.processed_label = None
        self.result_label = None
        self.confidence_label = None

        # Load model
        self.model = load_fashion_model()
        if self.model is None:
            messagebox.showerror("Error", "Could not load model")
            self.root.quit()
            return

        self.create_widgets()

    def create_widgets(self):
        # Title
        title_label = Label(self.root, text="Fashion Item Classifier",
                          font=("Arial", 18, "bold"), bg="#f0f0f0")
        title_label.pack(pady=15)

        # Frame for original image
        self.img_frame = Frame(self.root, bg="#f0f0f0")
        self.img_frame.pack(pady=10)

        # Original image display
        self.display_label = Label(self.img_frame, text="No image selected",
                                 width=28, height=10, bg="#e0e0e0")
        self.display_label.pack(side=tk.LEFT, padx=10)

        # Processed image display
        self.processed_label = Label(self.img_frame, text="Processed version",
                                   width=28, height=10, bg="#e0e0e0")
        self.processed_label.pack(side=tk.RIGHT, padx=10)

        # Upload button
        upload_btn = Button(self.root, text="Upload Image", command=self.upload_image,
                          bg="#4CAF50", fg="white", font=("Arial", 12))
        upload_btn.pack(pady=15)

        # Prediction frame
        pred_frame = Frame(self.root, bg="#f0f0f0")
        pred_frame.pack(pady=15, fill=tk.X)

        # Prediction result
        self.result_label = Label(pred_frame, text="", font=("Arial", 14, "bold"),
                                bg="#f0f0f0", fg="#0066cc")
        self.result_label.pack()

        # Confidence
        self.confidence_label = Label(pred_frame, text="", font=("Arial", 12),
                                    bg="#f0f0f0")
        self.confidence_label.pack(pady=5)

        # Instructions
        instructions = ("Upload a clothing image to classify into one of these categories:\n" +
                       ", ".join(CLASS_NAMES)+
                        "\n\nFor best results, use images with:\n" +
                        "• Clear, centered clothing items\n" +
                        "• Plain/empty backgrounds\n" +
                        "• Good lighting and contrast\n" +
                        "\nAccepted file types: JPG, JPEG, PNG, BMP, GIF" +
                        "\n\n\nThe model was trained on simple, standardized images.")
        instruction_label = Label(self.root, text=instructions, font=("Arial", 10),
                                bg="#f0f0f0", wraplength=550)
        instruction_label.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])

        if not file_path:
            return

        # Preprocess image
        img_array, img = preprocess_image(file_path)
        if img_array is None:
            self.result_label.config(text="Error processing image")
            return

        # Display original image
        display_img = Image.open(file_path)
        display_img = display_img.resize((140, 140))
        img_tk = ImageTk.PhotoImage(display_img)
        self.display_label.config(image=img_tk, text="")
        self.display_label.image = img_tk  # Keep reference

        # Display processed image (enlarged for visibility)
        processed_img = Image.fromarray((img_array[0, :, :, 0] * 255).astype(np.uint8))
        # Fix for NEAREST constant
        processed_img = processed_img.resize((140, 140), resample=Image.Resampling.NEAREST)
        proc_tk = ImageTk.PhotoImage(processed_img)
        self.processed_label.config(image=proc_tk, text="")
        self.processed_label.image = proc_tk  # Keep reference

        # Make prediction
        predictions = self.model.predict(img_array)[0]
        pred_class = np.argmax(predictions)
        confidence = predictions[pred_class] * 100

        # Create prediction text (removed unused top3_indices)
        result_text = f"Prediction: {CLASS_NAMES[pred_class]}"

        self.result_label.config(text=result_text)
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")

def main():
    try:
        root = tk.Tk()
        FashionClassifierApp(root)
        root.mainloop()
    except Exception as e:
        logger.error(f"Error in application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
