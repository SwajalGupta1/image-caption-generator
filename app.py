import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.models import Model


st.title("Image Caption Generator")
st.write("Upload an image to generate a caption.")
# Load the pre-trained model and tokenizer
model= load_model('mymodel2.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
# Load the VGG16 model for feature extraction
vgg_model = VGG16()
vgg16=Model(inputs=vgg_model.input, outputs=vgg_model.layers[-2].output)

def extract_features(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Extract features using VGG16
    features = vgg16.predict(img_array)
    return features

def generate_caption(image):
    # Extract features from the image
    features = extract_features(image)
    
    # Initialize the caption with the start token
    caption = ['startseq']

    # Generate caption word by word
    for _ in range(20):  # Limit to 20 words
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = np.array(sequence).reshape(1, -1)
        
        # Predict the next word
        yhat = model.predict([features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        
        # Convert the predicted index to a word
        word = tokenizer.index_word.get(yhat, None)
        
        if word is None or word == 'endseq':
            break
        
        caption.append(word)
    
    return ' '.join(caption[1:]) 

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:

    
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Generate and display the caption
    if st.button("Generate Caption"):
        caption = generate_caption(uploaded_file)
        st.write("Generated Caption:", caption)
    


