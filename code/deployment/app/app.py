# code/deployment/app/app.py
import streamlit as st
import requests
from PIL import Image
import io

st.title("Animal Classifier")
st.write("Upload an image to classify it as a cat, dog, or horse.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Convert image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()

    # Make prediction
    if st.button("Classify"):
        response = requests.post("http://api:8000/predict/", files={"file": img_bytes})
        prediction = response.json()['prediction']
        st.write(f"Prediction: **{prediction}**")
