import streamlit as st
import tensorflow as tf
import tweepy
import pickle
import sys
import os
import time
from pathlib import Path
from tweepy.errors import TooManyRequests

st.title("🌍  Emotion Classifier")

# language selection
language = st.selectbox("Select language:", ["English", "Spanish"])

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from data.preprocess import clean_text
except ImportError as e:
    st.error(f"Failed to import clean_text: {str(e)}")
    st.stop()

# emotion labels
labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 
          'optimism', 'pessimism', 'sadness', 'surprise', 'trust']

# model paths
BASE_DIR = Path(__file__).parent.parent
model_path = BASE_DIR / f"saved_models/{'english' if language == 'English' else 'spanish'}_model.keras"
tokenizer_path = BASE_DIR / f"data/{'tokenizer' if language == 'English' else 'spanishtokenizer'}.pkl"
max_len = 50

if not model_path.exists():
    st.error(f"Model file not found at: {model_path}")
    st.stop()

if not tokenizer_path.exists():
    st.error(f"Tokenizer file not found at: {tokenizer_path}")
    st.stop()

@st.cache_resource
def load_model_tokenizer(model_path, tokenizer_path):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load: {str(e)}")
        st.stop()

# load resources
model, tokenizer = load_model_tokenizer(model_path, tokenizer_path)

# authenticate
client = tweepy.Client(bearer_token="MY_TOKEN")

@st.cache_data(show_spinner=False)  # cache the tweet so it's not re-fetched
def get_tweet_text(tweet_url):
    time.sleep(5) 
    try:
        tweet_id = tweet_url.split('/')[-1]
        tweet = client.get_tweet(tweet_id, tweet_fields=["text"])
        return tweet.data['text']
    except TooManyRequests as e:
        reset_time = int(e.response.headers.get('x-rate-limit-reset'))  
        current_time = time.time() 
        sleep_time = reset_time - current_time   
        
        if sleep_time > 0:
            st.warning(f"Rate limit reached. Please wait {int(sleep_time)} seconds.")
            time.sleep(sleep_time)  
        
        return get_tweet_text(tweet_url) 
    except Exception as e:
        return f"Error: {str(e)}"

# input options
input_option = st.radio("Input method:", ["Enter text", "Provide Tweet URL"])

input_text = ""
tweet_url = ""

def fetch_and_analyze_tweet():
    global input_text
    tweet_url = st.session_state.tweet_url
    if tweet_url:
        if not tweet_url.startswith(('https://twitter.com/', 'https://x.com/')):
            st.warning("Please enter a valid Twitter/X URL")
        else:
            with st.spinner("Fetching tweet..."):
                input_text = get_tweet_text(tweet_url)
            if input_text:
                st.success("Successfully fetched tweet!")
                st.text_area("Extracted tweet:", value=input_text, height=100)
                
                # run model analysis when tweet is fetched
                try:
                    cleaned = clean_text(input_text)
                    seq = tokenizer.texts_to_sequences([cleaned])
                    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)
                    pred = model.predict(padded)[0]
                    pred_labels = [label for label, p in zip(labels, pred) if p > 0.5]

                    st.subheader("Detected Emotions:")
                    if pred_labels:
                        st.success(", ".join(pred_labels))
                    else:
                        st.info("No strong emotions detected.")
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

def analyze_text_input():
    global input_text
    if input_text:
        try:
            cleaned = clean_text(input_text)
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)
            pred = model.predict(padded)[0]
            pred_labels = [label for label, p in zip(labels, pred) if p > 0.5]

            st.subheader("Detected Emotions:")
            if pred_labels:
                st.success(", ".join(pred_labels))
            else:
                st.info("No strong emotions detected.")
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

# trigger analysis when user hits Enter
if input_option == "Enter text":
    with st.form(key="text_form"):
        input_text = st.text_area("Enter your tweet text:")
        submit = st.form_submit_button("Analyze Text")
        if submit:
            analyze_text_input()
else:
    st.text_input("Enter Tweet URL (e.g., https://twitter.com/user/status/123456789):", key="tweet_url", on_change=fetch_and_analyze_tweet)

