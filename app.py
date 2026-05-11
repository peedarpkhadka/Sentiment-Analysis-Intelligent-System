import streamlit as st
import tensorflow as tf 
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# set up page config
st.set_page_config(page_title="Sentiment Analysis Intelligent System", page_icon="🤖")

# Load assets (the brain and the translator)
@st.cache_resource         # This keeps the model in memory so it doesn't reload every time
def load_model_and_tokenizer():
    model = tf.keras.models.load_model('sentiment_model.h5')
    tokenizer = joblib.load('tokenizer.pkl')
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# app interface
st.title("🤖 Sentiment AI")
st.write("Type a book review below to see if the sentiment is Positive, Neutral, or Negative.")

user_input = st.text_area("Enter your review here: ", placeholder="I really enjoyed the characters in this book.......")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # convert the text to numbers
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=150, padding='post')

        # predict
        prediction = model.predict(padded, verbose=0)
        classes = ['Negative', 'Neutral', 'Positive']
        result = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # display result
        st.divider()
        if result == 'Positive':
            st.success(f"**Sentiment:** {result}")
        elif result == 'Neutral':
            st.info(f"**Sentiment:** {result}")
        else:
            st.error(f"**Sentiment:** {result}")

        st.write(f"**Confidence Score:** {confidence:.2f}%")
st.sidebar.info("This app use a TensorFlow Deep Learning model trained on 12,000 kindle review.")
