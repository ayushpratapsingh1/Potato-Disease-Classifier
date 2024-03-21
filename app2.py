import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2

# Load model and class indices
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "plant_disease_prediction_model.keras")
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

class_indices_path = os.path.join(working_dir, "class_indices.json")
try:
    with open(class_indices_path, "r") as f:
        class_indices = json.load(f)
except Exception as e:
    st.error(f"Error loading class indices: {e}")
    st.stop()

def load_and_preprocess_image(image, target_size=(224, 224)):
    if isinstance(image, np.ndarray):
        # Image is captured
        img_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # Image is uploaded
        img_array = np.array(image)

    # Resize the image
    img = Image.fromarray(img_array).resize(target_size)
    
    # Convert the image to a numpy array
    img_array = np.array(img)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    
    return img_array

def predict_image_class(model, image_array, class_indices):
    preprocessed_img = load_and_preprocess_image(image_array)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown")
    return predicted_class_name

# CSS styles
st.markdown(
    """
    <style>
        .reportview-container .main .block-container{
            max-width: 120%;
            padding-top: 1rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 1rem;
        }
        .reportview-container .main {
            color: black;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Logo and description
col1, col2 = st.columns([1, 3])
with col1:
    st.image("logo.png")
with col2:
    st.title('Plant Disease Classifier')
    st.write("An AI-powered app that helps identify plant diseases using images.")

# Camera button
class Camera(VideoTransformerBase):
    def transform(self, frame):
        return frame

if st.button('Capture Image'):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption='Webcam Image', use_column_width=True)

    # Predict using the captured image
    prediction = predict_image_class(model, frame, class_indices)

    # Release the camera
    cap.release()
    st.success(f'Prediction: {prediction}')

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image_array = np.array(image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Predict'):
            prediction = predict_image_class(model, image_array, class_indices)
            st.success(f'Prediction: {prediction}')
