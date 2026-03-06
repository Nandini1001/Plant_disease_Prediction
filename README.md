# Plant Disease Detection App
This is a web application that detects plant diseases from leaf images using a Convolutional Neural Network (CNN) model. The app allows users to upload a plant leaf image and predicts whether it is healthy or diseased, displaying the results in a clean and user-friendly interface built with Streamlit.
# Features
- Detects 38 plant disease classes (e.g., Apple, Corn, Tomato, Grape, Potato, etc.).

- Color-coded prediction:

✅ Green: Healthy

⚠️ Red: Diseased

- Clean UI.

- Sidebar with app instructions and model description.

# Project Structure
```
plant-disease-streamlit/
│
├─ app.py                 # Main Streamlit app
├─ model.h5               # Trained CNN model
├─ class_indices.json     # Class index mapping file
├─ requirements.txt       # Python dependencies
└─ README.md              # This file
```
# Usage
1)Run the Streamlit app:
'''
streamlit run app.py
'''
2)The app will open in your browser.

3)Upload a leaf image using the file uploader.

4)The app will display:

- Uploaded image

- Disease prediction (healthy/diseased)

# How to access Kaggle API Key
- Create your account on Kaggle
- Go to profile -> Settings
- Verify your account
- Click on Create Legacy API Key
- Click on Yes
- Kaggle.json file gets downloaded
- This file is needed while training the model

# About class_indices.json file
This file is not manually downloaded. It comes automatically on collab when you train the model and downloaded from collab.
