import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import imageio
import io
from PIL import Image

# Load your H5 model file
model = tf.keras.models.load_model("model.h5")

# Mapping of class indices to class names
CLASSES_LIST = ["Asthma", "Bandage", "Blood", "Blood Pressure", "Broke", "Burn", "Cold", "Constipated", "Cut", "Diarrhea", "Disease", "Doctor", "Emergency", "Fever", "Headache", "Infection", "Itch", "Nauseous", "Pain", "Patient", "Redness", "Sore Throat", "Urgent Care (UC)", "Anxiety", "Depressed", "Hospital", "Sprain", "Swallow", "Treatment", "Vomit"]

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100
SEQUENCE_LENGTH = 30

# Function to preprocess the frame and make predictions
def predict(frame):
    # Resize the frame to match the model's expected size
    img = image.load_img(io.BytesIO(frame), target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Expand the dimensions to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)

    # Perform prediction
    predictions = model.predict(img_array)

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
                # Read video frames and perform processing
                video_bytes = uploaded_file.read()
                video_reader = imageio.get_reader(io.BytesIO(video_bytes), 'pillow')
                predictions = []

                for frame in video_reader:
                    # Convert the frame to bytes for prediction
                    frame_bytes = imageio.imwrite(np.array(frame), format='JPEG').tobytes()

                    # Perform prediction on the frame
                    prediction = predict(frame_bytes)
                    predictions.append(prediction)

                # Display the prediction result after processing the entire video
                st.success(f"Predicted Class: {predictions[-1]}")

app()
