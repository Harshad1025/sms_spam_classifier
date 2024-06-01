import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from scipy.sparse import csr_matrix
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()

nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()

# Load spam image
spam_image = open("spam_image.png", "rb").read()

# Define emoji icons
spam_emoji = "🚨"
not_spam_emoji = "✅"

# @st.cache_data
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

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

initial_input_sms_value = ""  # Initial value for input text area

st.title("Email/SMS Spam Classifier")

# Unique key for the input text area
input_sms_key = "input_sms"

# Use a separate variable to store the value of the text area
input_sms_value = st.text_area("Enter the message", key=input_sms_key, value=initial_input_sms_value)

# Create an empty placeholder for displaying the output
output_placeholder = st.empty()

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms_value)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])  # Directly transform using tfidf
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        output_placeholder.error(f"{spam_emoji} Spam")
        st.image(spam_image, width=150)  # Display a smaller image
    else:
        output_placeholder.success(f"{not_spam_emoji} Not Spam")

# Add a home button
if st.button("Home"):
    input_sms_value = ''
