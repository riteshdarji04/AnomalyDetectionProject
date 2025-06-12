import streamlit as st
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer
from PIL import Image

# Title
st.set_page_config(page_title="Anomaly Detection App", layout="centered")
st.title("🔍 AI-Based Anomaly Detection System")

# Load model using TFSMLayer (Teachable Machine's SavedModel)
@st.cache_resource
def load_model():
    return TFSMLayer("model", call_endpoint="serving_default")

model = load_model()

# File uploader UI
uploaded_file = st.file_uploader("📤 Upload an image of the product", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Perform prediction
    output = model(img_array)  # This returns a dictionary
    predictions = list(output.values())[0].numpy()[0]  # Get the array

    class_names = ["Normal", "Anomaly"]

    # Display prediction probabilities
    st.subheader("🧠 Prediction Probabilities:")
    for i, prob in enumerate(predictions):
        st.write(f"🔹 {class_names[i]}: **{prob * 100:.2f}%**")

    # Final Prediction
    predicted_class = class_names[np.argmax(predictions)]
    st.success(f"✅ Final Prediction: **{predicted_class}**")
