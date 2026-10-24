import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("Email Spam & Phishing Detection System")
st.write("Enter email text to check if it is Spam or Ham.")

user_input = st.text_area("Email Text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to check.")
    else:
        # Transform input text
        input_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(input_vectorized)
        
        if prediction[0] == 1:
            st.error("🚨 Warning: This is a SPAM / Phishing Email!")
        else:
            st.success("✅ Safe: This is a Normal (Ham) Email!")