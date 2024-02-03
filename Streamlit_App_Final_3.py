from PIL import Image as PilImage
import streamlit as st
import tensorflow as tf
import requests
from io import BytesIO

# Function to preprocess the image
def preprocess_image(image_path):
    pil_image = PilImage.open(image_path)
    pil_image_resized = pil_image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(pil_image_resized)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array /= 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Function to download the model from GitHub using Git LFS
def download_model_from_github(model_url, model_filename):
    response = requests.get(model_url)
    model_content = BytesIO(response.content)
    with open(model_filename, 'wb') as f:
        f.write(model_content.getvalue())

# Main Streamlit code
st.title("Dog vs Cat Image Classifier")

# Upload an image through Streamlit
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        pil_image = PilImage.open(uploaded_file)
        st.image(pil_image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the resized image for the model
        img_array = preprocess_image(uploaded_file)

        # Download the model from GitHub using Git LFS
        model_url = 'https://raw.githubusercontent.com/Smail-Bel/test_1/main/dog_cat_detector_model_Final_2.h5'
        model_filename = 'dog_cat_detector_model_Final_2.h5'
        download_model_from_github(model_url, model_filename)

        # Load the pre-trained model
        model = tf.keras.models.load_model(model_filename)

        # Make predictions using the loaded model
        predictions = model.predict(img_array)

        # Print raw predictions
        st.write("Raw Predictions:", predictions[0].tolist())

        # Manually interpret predictions based on your model's output with a threshold
        threshold = 0.5
        predicted_class = 'Dog' if predictions[0][0] >= threshold else 'Cat'
        confidence = predictions[0][0]

        # Display the result
        st.subheader("Prediction:")
        st.write(f"Predicted class: {predicted_class}")
        st.write(f"Confidence: {float(confidence):.2%}")

    except Exception as e:
        st.error(f"Error processing the image: {e}")
