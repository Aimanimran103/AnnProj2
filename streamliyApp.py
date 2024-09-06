import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Digit Recognition with ANN")

# Load your trained model
model = tf.keras.models.load_model('path_to_your_ann_model.h5')

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image before prediction
    img_array = np.array(image.resize((28, 28)))  # Adjust size according to your model
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Reshape for prediction

    # Make prediction
    prediction = model.predict(img_array)
    st.write(f"Predicted digit: {np.argmax(prediction)}")

