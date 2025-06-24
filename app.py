import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

st.title("Image Caption Generator")
st.write("Upload an image to generate a caption.")

# Load the pre-trained model and tokenizer
model = load_model('mymodel2.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the VGG16 model for feature extraction
vgg_model = VGG16()
vgg16 = Model(inputs=vgg_model.input, outputs=vgg_model.layers[-2].output)

def extract_features(image_file):
    # Open and preprocess the uploaded image
    img = Image.open(image_file).resize((224, 224)).convert("RGB")
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Extract features using VGG16
    features = vgg16.predict(img_array)
    return features

def generate_caption(image_file):
    features = extract_features(image_file)
    caption = ['startseq']
    for _ in range(20):  # Max 20 words
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = np.array(sequence).reshape(1, -1)

        yhat = model.predict([features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)

        if word is None or word == 'endseq':
            break
        caption.append(word)

    return ' '.join(caption[1:])  # Exclude 'startseq'

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Generate Caption"):
        caption = generate_caption(uploaded_file)
        st.write("Generated Caption:", caption)
