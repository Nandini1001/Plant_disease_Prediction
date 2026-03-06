import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import json

# Load model
model = load_model('model.h5')

# Load class indices from JSON
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
    
# Convert JSON to a list where index matches model output
classes = [class_indices[str(i)] for i in range(len(class_indices))]

# Streamlit page config
st.set_page_config(page_title="Plant Disease Detection 🌱", page_icon="🌿", layout="wide")

# Gradient background CSS
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #a8e6cf, #dcedc1, #ffd3b6);
        background-attachment: fixed;
    }
    .stApp {
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 8em;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("🌿 Plant Disease Detection App")
st.sidebar.info(
    """
    Upload a leaf image and the app will detect its disease using a trained CNN model.

    - ✅ Green: Healthy
    - ⚠️ Red: Diseased

    **Instructions:**
    1. Click 'Browse files' to upload an image of a plant leaf.
    2. Wait for the prediction to appear below.
    3. Ensure the image is clear and focused for best results.
    """
)

# Title
st.title("🌱 Plant Disease Detection")
st.write("Upload a leaf image to detect its disease and see class probabilities.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Leaf', width=400)

    # Preprocess image
    img_resized = img.resize((224, 224))
    x = image.img_to_array(img_resized)/255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    pred = model.predict(x)           # shape: (1, num_classes)
    pred = pred.flatten()             # flatten to 1D array
    if len(pred) != len(classes):
        st.error(f"Model output ({len(pred)}) does not match number of classes ({len(classes)})")
    else:
        pred_class = classes[np.argmax(pred)]
        pred_prob = np.max(pred)

        # Display prediction
        if 'healthy' in pred_class.lower():
            st.markdown(
                f"<h2 style='color:green; text-align:center;'>✅ Prediction: {pred_class} ({pred_prob*100:.2f}%)</h2>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<h2 style='color:red; text-align:center;'>⚠️ Prediction: {pred_class} ({pred_prob*100:.2f}%)</h2>",
                unsafe_allow_html=True
            )
