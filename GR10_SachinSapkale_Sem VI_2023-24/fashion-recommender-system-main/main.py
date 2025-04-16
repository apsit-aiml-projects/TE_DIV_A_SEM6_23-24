import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import pickle

# Load precomputed features and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Create a new model for feature extraction
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Function to extract features from an image
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Function to recommend unique images
def recommend_unique(features, feature_list, shown_indices):
    neighbors = NearestNeighbors(n_neighbors=len(feature_list), algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    # Find nearest neighbors excluding shown indices
    distances, indices = neighbors.kneighbors([features])
    new_recommendations = [idx for idx in indices[0] if idx not in shown_indices][:5]
    
    return new_recommendations

# Streamlit app
st.title('Fashion Recommender System')

uploaded_file = st.file_uploader("Choose an image")

if uploaded_file is not None:
    if st.button("Generate Recommendations"):
        # Save uploaded image to a temporary directory
        temp_image_path = os.path.join('temp', uploaded_file.name)
        with open(temp_image_path, 'wb') as f:
            f.write(uploaded_file.read())
        
        # Display uploaded image
        st.image(Image.open(temp_image_path))

        # Extract features from the uploaded image
        features = feature_extraction(temp_image_path, model)

        # Get previously shown indices from session state
        shown_indices = st.session_state.get('shown_indices', [])

        # Generate new unique recommendations
        new_recommendations = recommend_unique(features, feature_list, shown_indices)

        if len(new_recommendations) == 0:
            st.info("No more unique recommendations available.")
        else:
            # Display recommended images
            st.write("Recommended Images:")
            cols = st.columns(len(new_recommendations))
            for i, col in enumerate(cols):
                recommended_image = Image.open(filenames[new_recommendations[i]]).resize((120, 120))
                col.image(recommended_image, caption=f"Recommended {i+1}", use_column_width=False)
                
            # Update shown indices in session state
            st.session_state['shown_indices'] = shown_indices + new_recommendations
