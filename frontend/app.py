import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.preprocess import clean_text

# Load model and tokenizer
import tensorflow as tf
model = tf.keras.models.load_model("saved_models/english_model.keras", compile=False)
with open("data/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 
          'optimism', 'pessimism', 'sadness', 'surprise', 'trust']

st.title("🌍 Emotion Classifier")
text_input = st.text_area("Enter a tweet:")

if st.button("Analyze"):
    if text_input:
        cleaned = clean_text(text_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)

        pred = model.predict(padded)[0]
        pred_labels = [label for label, p in zip(labels, pred) if p > 0.5]

        st.subheader("Detected Emotions:")
        if pred_labels:
            st.write(", ".join(pred_labels))
        else:
            st.write("No strong emotions detected.")