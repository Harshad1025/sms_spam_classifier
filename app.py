import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Porter Stemmer
ps = PorterStemmer()

# Load spam image
spam_image = open("spam_image.png", "rb").read()

# Define emoji icons
spam_emoji = "🚨"
not_spam_emoji = "✅"

# Load the trained model and vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Streamlit UI
st.title("Email/SMS Spam Classifier")

# Input text area
input_sms_value = st.text_area("Enter the message")

# Display predictions
if st.button('Predict'):
    transformed_sms = transform_text(input_sms_value)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.error(f"{spam_emoji} Spam")
        st.image(spam_image, width=150)
    else:
        st.success(f"{not_spam_emoji} Not Spam")
