# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    # Vocabulary size used during training
    MAX_VOCAB_SIZE = 10000
    
    words = text.lower().split()
    encoded_review = []
    for word in words:
        # Get the word index, if not found use 2 (unknown token)
        idx = word_index.get(word, 2)
        # Add 3 to match IMDB dataset preprocessing
        idx = idx + 3
        # If index exceeds vocabulary size, use unknown token
        if idx >= MAX_VOCAB_SIZE:
            idx = 2 + 3  # unknown token
        encoded_review.append(idx)
    
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# Import Streamlit at the top level
import streamlit as st

# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    # Preprocess the input
    preprocessed_input = preprocess_text(user_input)
    
    # Make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')

