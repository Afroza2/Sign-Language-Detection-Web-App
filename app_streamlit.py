import streamlit as st
import requests

st.title("Sign Language Prediction App")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Predict"):
        with st.spinner("Processing..."):
            try:
                # Read video frames and prepare for prediction
                video_content = uploaded_file.read()

                # Perform prediction on video frames
                response = requests.post("http://localhost:8000/predict-video/", files={"video_file": video_content})

                # print("response", response.status_code)
                
                if response.status_code == 200:
                    predictions = response.json().get("predicted_class", "Prediction not available")  # Fix key here
                    # confidence = response.json().get("confidence", 0.0)  # Get confidence score
                    st.success(f"Predicted Class: {predictions}\n")
                else:
                    st.error("Error during prediction. Please try again.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
