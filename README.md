# Emotion Sense: Multilingual Tweet Emotion Detection

**Team Members:** Alexa Tallo, Crystal Shamo, and Miray Noeel  
**Course:** CSI 5130 – Artificial Intelligence  
**Instructor:** Dr. Tianle Ma 

---

## Overview

**Emotion Sense** is a multilingual web application designed to detect emotions in tweets in real time. Built with Streamlit, the app allows users to input a tweet URL or plain text and receive predictions of emotional labels using models trained on the SemEval-2018 Task 1: Affect in Tweets dataset.

Our system supports both **English** and **Spanish** tweets and includes the following models
- CNN (English)
- CNN (Spanish)

We experimented with various models and found that the **Convolutional Neural Network (CNN)** delivered the best performance, particularly in detecting rare emotions. As a result, **we selected the CNN as the final deployed model** in the application.

---

## Features

- Multi-label emotion classification (11 emotions)
- English and Spanish language support
- Real-time predictions via a web UI (Streamlit)
- Four models available for evaluation and comparison
- Optimized CNN with oversampling, class weighting, and threshold tuning

---

## Supported Emotions

- anger  
- anticipation  
- disgust  
- fear  
- joy  
- love  
- optimism  
- pessimism  
- sadness  
- surprise  
- trust  

---

## 🎥 Demo Video

Experience Emotion Sense in action:  
🔗 [Watch the demo on YouTube](https://youtu.be/ifvlHzxynIc?si=eFmWFk5BtRVpuVdB) 

---

## 🖥️ App Demo

To run the app locally:

  Note: The Twitter Bearer Token has been removed for security reasons. 

```bash
git clone https://github.com/alexatallo/emotion-detection-tweets.git
cd emotion-sense
pip install -r requirements.txt
streamlit run app.py

