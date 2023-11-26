import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
import tempfile
import io

# Load your H5 model file
model = tf.keras.models.load_model("model.h5")

# Mapping of class indices to class names
CLASSES_LIST = ["Asthma", "Bandage", "Blood", "Blood Pressure", "Broke", "Burn", "Cold", "Constipated", "Cut", "Diarrhea", "Disease", "Doctor", "Emergency", "Fever", "Headache", "Infection", "Itch", "Nauseous", "Pain", "Patient", "Redness", "Sore Throat", "Urgent Care (UC)", "Anxiety", "Depressed", "Hospital", "Sprain", "Swallow", "Treatment", "Vomit"]

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100
SEQUENCE_LENGTH = 30

# Function to preprocess the frame and make predictions
def predict(frame):
    # Repeat the frame to create a sequence of frames
    frames_sequence = np.repeat(np.expand_dims(frame, axis=0), SEQUENCE_LENGTH, axis=0)

    # Resize the frames to match the model's expected size
    resized_frames = [image.img_to_array(image.load_img(io.BytesIO(f), target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))) for f in frames_sequence]

    # Stack the resized frames along the time axis
    input_sequence = np.stack(resized_frames, axis=0)

    # Perform prediction
    predictions = model.predict(np.expand_dims(input_sequence, axis=0))

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions)

    # Get the predicted class name
    predicted_class = CLASSES_LIST[predicted_class_index]

    return predicted_class

# Streamlit app
st.title("Sign Language Prediction App")

# Function to display video stream and predictions
def app():
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

    if uploaded_file is not None:
        st.video(uploaded_file)

        if st.button("Predict"):
            with st.spinner("Processing..."):
                # Save the uploaded file to a temporary location
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(uploaded_file.read())

                # Read video frames and perform processing
                cap = cv2.VideoCapture(temp_file.name)
                predictions = []

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Convert the frame to bytes for prediction
                    frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()

                    # Perform prediction on the frame
                    prediction = predict(frame_bytes)
                    predictions.append(prediction)

                cap.release()

                # Display the prediction result after processing the entire video
                st.success(f"Predicted Class: {predictions[-1]}")

app()
