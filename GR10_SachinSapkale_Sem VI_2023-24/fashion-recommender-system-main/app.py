import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# Load the pre-trained ResNet50 model without top classification layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add global pooling to extract fixed-size features
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Function to extract normalized features from image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Collect image paths
filenames = [os.path.join('images1', file) for file in os.listdir('images1')]

# Extract features for all images
feature_list = [extract_features(file, model) for file in tqdm(filenames)]

# Save features and filenames to disk
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))
