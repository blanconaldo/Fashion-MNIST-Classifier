import numpy as np
import streamlit as st
from keras.src.saving import load_model
from PIL import Image
import logging
import pandas as pd
import matplotlib.pyplot as plt
import base64
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem !important;
        color: #FF4B4B;
        text-align: center;
    }
    .subheader {
        font-size: 2.0rem !important;
        color: #4B4BFF;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .prediction-box {
        background-color: #e6f7ff;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #1890ff;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid #ddd;
        color: #666;
        font-size: 0.9rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Class names from Fashion MNIST
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Class descriptions for more detailed explanations
CLASS_DESCRIPTIONS = {
    "T-shirt/top": "A casual upper-body garment with short sleeves and no collar.",
    "Trouser": "A garment that covers the lower body from waist to ankles, with separate sections for each leg.",
    "Pullover": "A knitted garment that covers the upper body and is pulled over the head to be worn.",
    "Dress": "A one-piece garment consisting of a top and skirt, typically worn by women or girls.",
    "Coat": "A heavy outer garment worn for protection against cold weather.",
    "Sandal": "An open footwear with straps attaching the sole to the foot.",
    "Shirt": "A more formal, collared upper-body garment with buttons down the front.",
    "Sneaker": "A comfortable, casual athletic shoe with a flexible sole.",
    "Bag": "A container made of flexible material with an opening at the top, used for carrying items.",
    "Ankle boot": "A short boot that covers the foot and ankle, but not the calf."
}

# Load model once using Streamlit caching
@st.cache_resource
def load_fashion_model(model_path="./models/fashion_mnist_model_best.keras"):
    try:
        with st.spinner("Loading model... (first run only)"):
            logger.info(f"Loading model from {model_path}")
            model = load_model(model_path)
            logger.info("Model loaded successfully")
            return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    try:
        # Convert to grayscale
        img = image.convert('L')

        # Resize to 28x28
        img = img.resize((28, 28))

        # Convert to numpy array and normalize
        img_array = np.array(img)
        img_array = img_array / 255.0

        # Invert colors if needed
        if np.mean(img_array) > 0.5:
            img_array = 1 - img_array

        # Force consistent data type to prevent retracing
        img_array = img_array.astype(np.float32)

        # Expand dimensions to match model input shape
        img_array = img_array.reshape(1, 28, 28, 1)

        return img_array, img
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None, None

def get_confidence_color(confidence):
    """Return color based on confidence level"""
    if confidence >= 90:
        return "green"
    elif confidence >= 70:
        return "orange"
    else:
        return "red"

def main():
    try:
        # Sidebar elements
        with st.sidebar:
            st.image("https://miro.medium.com/v2/resize:fit:1400/1*QQVbuP2SEasB0XAmvjW0AA.jpeg",
                    caption="Fashion MNIST Examples")

            st.markdown("### About")
            st.info(
                "This app uses a Convolutional Neural Network trained on the "
                "Fashion MNIST dataset to classify clothing items from uploaded images."
            )

            st.markdown("### Model Info")
            st.write("CNN Architecture: 4 Conv layers, 2 Dense layers")
            st.write("Accuracy: 92.53% on test data")

            st.markdown("### How It Works")
            st.write("1. Upload an image")
            st.write("2. Image is converted to grayscale & resized")
            st.write("3. Neural network predicts the clothing type")
            st.write("4. Results are displayed with confidence")

            # Add history section
            if 'history' in st.session_state:
                st.markdown("### Recent Predictions")
                for i, item in enumerate(st.session_state.history[-3:]):
                    st.write(f"{i+1}. {item['class']} ({item['confidence']:.1f}%)")

        # Initialize session state for history
        if 'history' not in st.session_state:
            st.session_state.history = []

        # Main content header
        st.markdown("<h1 class='main-header'>Fashion Item Classifier</h1>", unsafe_allow_html=True)

        # Tab navigation
        tab1, tab2, tab3 = st.tabs(["Classifier", "Examples Gallery", "About the Model"])

        with tab1:
            # Load model once
            model = load_fashion_model()
            if model is None:
                st.error("Could not load model. Please refresh the page or check logs.")
                return

            # Instructions
            with st.container():
                st.markdown("<h2 class='subheader'>Instructions</h2>", unsafe_allow_html=True)
                st.write("Upload a clothing image to classify into one of these categories:")
                categories_html = ", ".join([f"<b>{name}</b>" for name in CLASS_NAMES])
                st.markdown(f"{categories_html}", unsafe_allow_html=True)

            # File uploader with better styling
            uploaded_file = st.file_uploader("Choose an image...",
                                        type=["jpg", "jpeg", "png", "bmp", "gif"])

            col1, col2 = st.columns([1, 1])

            # Show tips in the first column
            with col1:
                with st.expander("Tips for best results", expanded=False):
                    st.markdown("""
                    ### For best results, use images with:
                    - Clear, centered clothing items
                    - Plain/empty backgrounds
                    - Good lighting and contrast

                    ### Accepted file types:
                    JPG, JPEG, PNG, BMP, GIF

                    The model was trained on simple, standardized images.
                    """)

            # Display layout
            if uploaded_file is not None:
                # Read image
                image = Image.open(uploaded_file)

                # Create two columns for original and processed images
                img_col1, img_col2 = st.columns(2)

                with img_col1:
                    st.markdown("<h3 class='subheader'>Original Image</h3>", unsafe_allow_html=True)
                    st.image(image, width=250, caption="Your uploaded image")

                # Preprocess image
                img_array, processed_img = preprocess_image(image)

                if img_array is not None:
                    # Display processed image
                    with img_col2:
                        st.markdown("<h3 class='subheader'>Processed Image</h3>", unsafe_allow_html=True)
                        # Resize for better visibility
                        display_img = Image.fromarray((img_array[0, :, :, 0] * 255).astype(np.uint8))
                        display_img = display_img.resize((140, 140), resample=Image.Resampling.NEAREST)
                        st.image(display_img, width=250, caption="As seen by the model (28x28)")

                    # Add a predict button
                    if st.button("Predict Now!", key="predict_button"):
                        # Show spinner during prediction
                        with st.spinner("Analyzing your image..."):
                            time.sleep(0.5)  # Add small delay for UX
                            # Make prediction
                            predictions = model.predict(img_array, verbose=0)[0]
                            pred_class = np.argmax(predictions)
                            confidence = float(predictions[pred_class] * 100)  # Convert to Python float

                            # Add to history
                            st.session_state.history.append({
                                'class': CLASS_NAMES[pred_class],
                                'confidence': confidence
                            })

                        # Display prediction in a nice container
                        st.markdown(f"### Prediction: {CLASS_NAMES[pred_class]}")
                        st.markdown(f"_{CLASS_DESCRIPTIONS[CLASS_NAMES[pred_class]]}_")

                        # Confidence display with color
                        conf_color = get_confidence_color(confidence)
                        st.markdown(f"<h3 style='color:{conf_color}'>Confidence: {confidence:.1f}%</h3>",
                                   unsafe_allow_html=True)

                        # Create progress bars for top 3 predictions
                        st.markdown("### Top Predictions")
                        top_indices = np.argsort(predictions)[-3:][::-1]

                        for i in top_indices:
                            conf = float(predictions[i] * 100)  # Convert to Python float
                            st.progress(conf/100)
                            st.write(f"{CLASS_NAMES[i]}: {conf:.1f}%")

                        # Create and display bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        top_classes = [CLASS_NAMES[i] for i in top_indices]
                        top_probs = [float(predictions[i] * 100) for i in top_indices]  # Convert to Python float

                        bars = ax.bar(top_classes, top_probs, color=['#1890ff', '#73d13d', '#ffc53d'])
                        ax.set_ylabel('Confidence (%)')
                        ax.set_title('Top 3 Predictions')

                        # Add percentage labels on top of bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.annotate(f'{height:.1f}%',
                                        xy=(bar.get_x() + bar.get_width() / 2, height),
                                        xytext=(0, 3),  # 3 points vertical offset
                                        textcoords="offset points",
                                        ha='center', va='bottom', fontweight='bold')

                        st.pyplot(fig)

                        # Add download button for results
                        result_data = pd.DataFrame({
                            'Class': [CLASS_NAMES[i] for i in range(10)],
                            'Confidence': [float(predictions[i] * 100) for i in range(10)]  # Convert to Python float
                        }).sort_values('Confidence', ascending=False)

                        csv = result_data.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a download="prediction_results.csv" href="data:file/csv;base64,{b64}">Download Results as CSV</a>'
                        st.markdown(href, unsafe_allow_html=True)
                else:
                    st.error("Error processing image. Please try a different image.")

        with tab2:
            st.markdown("<h2 class='subheader'>Example Fashion Items</h2>", unsafe_allow_html=True)
            st.write("Here are examples of each category the model was trained on:")

            # Display example grid
            example_images = {
                "T-shirt/top": "./cached_images/T-shirt-Top.jpg",
                "Trouser": "./cached_images/Trouser.jpg",
                "Pullover": "./cached_images/Pullover.jpg",
                "Dress": "./cached_images/Dress.jpg",
                "Coat": "./cached_images/Coat.jpg",
                "Sandal": "./cached_images/Sandal.jpg",
                "Shirt": "./cached_images/Shirt.jpg",
                "Sneaker": "./cached_images/Sneaker.jpg",
                "Bag": "./cached_images/Bag.jpg",
                "Ankle boot": "./cached_images/Ankle_Boot.jpg"
            }

            # Display in 2 rows of 5
            row1_cols = st.columns(5)
            row2_cols = st.columns(5)

            for i, (class_name, img_url) in enumerate(example_images.items()):
                col = row1_cols[i % 5] if i < 5 else row2_cols[i % 5]
                with col:
                    st.image(img_url, width=120, caption=class_name)
                    st.write(CLASS_DESCRIPTIONS[class_name][:50] + "...")

            st.info("Example images sourced from Gap.com belong to Gap Inc. and they are showed for demonstration purposes only. All images are copyright of their respective owners.")
            st.info("Note: The model was trained on much simpler grayscale versions of these items. The items were of course varying in shape and color but this is just a simple visualization of the average item shape.")


        with tab3:
            st.markdown("<h2 class='subheader'>About the Model</h2>", unsafe_allow_html=True)

            # Display model architecture
            with st.container():
                st.markdown("### CNN Architecture")
                st.code("""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 32)        320

 batch_normalization         (None, 28, 28, 32)        128

 activation (Activation)     (None, 28, 28, 32)        0

 conv2d_1 (Conv2D)           (None, 28, 28, 32)        9248

 batch_normalization_1       (None, 28, 28, 32)        128

 activation_1 (Activation)   (None, 28, 28, 32)        0

 max_pooling2d               (None, 14, 14, 32)        0

 dropout (Dropout)           (None, 14, 14, 32)        0

 conv2d_2 (Conv2D)           (None, 14, 14, 64)        18496

 batch_normalization_2       (None, 14, 14, 64)        256

 activation_2 (Activation)   (None, 14, 14, 64)        0

 conv2d_3 (Conv2D)           (None, 14, 14, 64)        36928

 batch_normalization_3       (None, 14, 14, 64)        256

 activation_3 (Activation)   (None, 14, 14, 64)        0

 max_pooling2d_1             (None, 7, 7, 64)          0

 dropout_1 (Dropout)         (None, 7, 7, 64)          0

 flatten (Flatten)           (None, 3136)              0

 dense (Dense)               (None, 512)               1606144

 batch_normalization_4       (None, 512)               2048

 activation_4 (Activation)   (None, 512)               0

 dropout_2 (Dropout)         (None, 512)               0

 dense_1 (Dense)             (None, 10)                5130

=================================================================
Total params: 1,679,082
Trainable params: 1,677,674
Non-trainable params: 1,408
            """)

            # Performance metrics
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Training Process")
                st.write("- Dataset: 60,000 training images, 10,000 test images")
                st.write("- Resolution: 28x28 pixels grayscale")
                st.write("- Data Augmentation: Rotation, shifts, zooms")
                st.write("- Batch Size: 64")
                st.write("- Optimizer: Adam (lr=0.001)")
                st.write("- Early Stopping: Patience of 5 epochs")

            with col2:
                st.markdown("### Performance")
                st.write("- Test Accuracy: 92.53%")
                st.write("- Best performing class: Trousers (99%)")
                st.write("- Most challenging class: Shirts (78%)")
                st.write("- Training time: ~5 minutes on GPU")
                st.write("- Model size: ~20MB")
                st.write("- Prediction time: <100ms per image")

            # Confusion matrix visualization (fake data for UI mockup)
            with st.container():
                st.markdown("### Confusion Matrix")
                st.image("plots/Confusion_Matrix.png", use_container_width=True,
                    caption="Confusion matrix of model")

        # Footer
        st.markdown("<div class='footer'>", unsafe_allow_html=True)
        st.markdown("Fashion MNIST Classifier | Created for Portfolio Project | Model Accuracy: 92.53% | Loss: 0.2060")
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"Error in application: {e}")
        st.error(f"An unexpected error occurred: {e}")
        st.info("Please try refreshing the page or uploading a different image.")

if __name__ == "__main__":
    main()