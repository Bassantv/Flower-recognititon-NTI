import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Load your trained model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("flower model.keras", compile=False)

model = load_model()

# Define class names (adjust to your dataset)
class_names = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]

# -----------------------------
# Sidebar instructions
# -----------------------------
st.sidebar.title("📌 Instructions")
st.sidebar.markdown("""
1. Upload a flower image (`.jpg` / `.png`).  
2. The model will predict which flower it is.  
3. Results will appear on the main screen.  
""")

# -----------------------------
# Main app
# -----------------------------
st.title("🌸 Flower Recognition App")
st.write("Upload an image of a flower, and the model will predict its class.")

# File uploader
uploaded_file = st.file_uploader("Upload a flower image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_resized = image.resize((128, 128))  # match model input size
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]) * 100)

    # Show result
    st.subheader("Prediction Result")
    st.write(f"**Flower Type:** {class_names[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}%")


