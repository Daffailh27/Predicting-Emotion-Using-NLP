import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

# Initialize lemmatizer and stop words list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase, lemmatize, and remove stopwords
    words = [lemmatizer.lemmatize(word) for word in text.lower().split() if word not in stop_words and word.isalnum()]
    return ' '.join(words)

def predict_tfidf(text, model_path, tfidf_vectorizer_path):
    with open(tfidf_vectorizer_path, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    
    model = load_model(model_path)
    
    preprocessed_text = preprocess_text(text)
    x_new_tfidf = tfidf_vectorizer.transform([preprocessed_text])
    predictions = model.predict(x_new_tfidf.toarray())
    
    # Assuming you have a way to map predictions back to labels
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return predicted_class_index

# Define the category names mapping
category_names = {
    0: 'anger',
    1: 'fear',
    2: 'joy', 
    3: 'sadness',
    4: 'surprise',
    5: 'love'
}

def run():
    # Streamlit app setup
    st.title("Text Input for Prediction")

    user_input = st.text_area("Enter your text here:")

    if st.button('Predict'):
        model_path = 'my_trained_model.h5'  # Adjust path
        tfidf_vectorizer_path = 'tfidf_vectorizer.pkl'
        
        predicted_class_index = predict_tfidf(user_input, model_path, tfidf_vectorizer_path)
        
        # Display the predicted category
        st.write(f'Predicted Category: {category_names[predicted_class_index]}')

if __name__ == '__main__':
    run()
