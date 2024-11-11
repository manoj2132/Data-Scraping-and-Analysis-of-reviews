import pandas as pd  
import numpy as np  
import re  
import pickle  
import nltk  
import streamlit as st  
from nltk.tokenize import word_tokenize  
from nltk.corpus import stopwords  
from nltk.stem import WordNetLemmatizer  
from sklearn.feature_extraction.text import CountVectorizer  
from xgboost import XGBClassifier  
from textblob import TextBlob  

# Download necessary NLTK data only once  
nltk.download('stopwords', quiet=True)  
nltk.download('wordnet', quiet=True)  
nltk.download('punkt', quiet=True)  
nltk.download('omw-1.4', quiet=True)  

# Set up stop words and lemmatizer  
stop_words = set(stopwords.words('english'))  
lemmatizer = WordNetLemmatizer()  

# Streamlit app title and description  
st.title('🌟 NLP Sentiment Analysis Prediction 🌟')  
st.write('This app predicts if entered text contains positive, negative, or neutral feelings.')  

# User input with placeholder  
user_input = st.text_area("Enter some text:", height=150, placeholder="Type your text here...")  

# Function to clean text  
def clean_text(text):  
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphanumeric characters  
    text = text.lower()  # Lowercase  
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace  
    tokens = word_tokenize(text)  # Tokenize  
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1] #Remove stop words and single-letter words  
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize  
    return ' '.join(tokens)  

# Function to get sentiment  (Improved accuracy)  
def get_sentiment(text):  
    analysis = TextBlob(text)  
    polarity = analysis.sentiment.polarity  
    subjectivity = analysis.sentiment.subjectivity #added subjectivity for better understanding  

    if polarity > 0.2: #Increased threshold for positive sentiment  
        return 'positive', subjectivity  
    elif polarity < -0.2: #Increased threshold for negative sentiment  
        return 'negative', subjectivity  
    else:  
        return 'neutral', subjectivity  

# Function to highlight words  
def highlight_text(text, sentiment):  
    if sentiment == 'positive':  
        color = 'lightgreen'  
    elif sentiment == 'negative':  
        color = 'lightcoral'  
    else:  
        color = 'lightgray'  
    return f'<span style="background-color: {color};">{text}</span>'  

# Streamlit button with custom styling  
if st.button('🔍 Analyze', key='analyze'):  
    if user_input:  
        cleaned_text = clean_text(user_input)  
        sentiment, subjectivity = get_sentiment(cleaned_text)  
        highlighted_text = highlight_text(user_input, sentiment)  

        # Display results  
        st.subheader('Sentiment Analysis Result')  
        st.markdown(f'The sentiment of the entered text is: **{sentiment}**')  
        st.markdown(f"Subjectivity: {subjectivity:.2f}") # Display subjectivity score  
        st.markdown(f'Highlighted Text: {highlighted_text}', unsafe_allow_html=True)  


# Add a footer  
st.markdown("---")  
st.markdown("Developed by EXCELR 6th Team")  

# Sidebar with additional information  
st.sidebar.title("About")  
st.sidebar.info(  
    """  
    This app uses TextBlob for sentiment analysis.  
    Enter any text in the box to see if it is positive, negative, or neutral.  
    """  
)  
background_style = """  
<style>  
body {  
    background-image: url('your_image.jpg'); /* Replace with your image path */  
    background-size: cover;  
    background-repeat: no-repeat;  
    color: white; /* Adjust text color for contrast */  
    font-family: 'Arial', sans-serif;  
}  
</style>""" 
st.markdown(background_style, unsafe_allow_html=True)
# Custom CSS for enhanced styling (Improved and more robust)  
st.markdown(  
    """  
    <style>  
    body {  
        background: linear-gradient(to bottom right, #ffafbd, #ffc3a0); /* Simpler gradient */  
        color: #333;  
        font-family: 'Arial', sans-serif;  
    }  
    .stButton > button {  
        background-color: #4CAF50;  
        color: white;  
        border-radius: 8px;  
        padding: 10px 20px;  
        font-size: 16px;  
    }  
    .stTextArea {  
        border-radius: 8px;  
        border: 1px solid #ccc;  
        font-size: 16px; /* Added font size for better readability */  
        padding: 10px; /* Added padding for better look */  

    }  
    h1, h2, h3 {  
        color: #5c5c5c;  
    }  
    </style>  
    """,  
    unsafe_allow_html=True  
)