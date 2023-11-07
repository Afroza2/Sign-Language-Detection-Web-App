# fastapi_app.py

from fastapi import FastAPI, UploadFile, File
from starlette.responses import JSONResponse
import tensorflow.keras.models
import numpy as np
import cv2
import tempfile
import os

app = FastAPI()

# Load your Keras model
model_path = 'Mobi_LRCN_model_LSTM_128_Date_Time_2023_10_31__22_51_35___Loss_0.3103679418563843___Accuracy_0.9332405924797058.h5'
model = tensorflow.keras.models.load_model(model_path)

CLASSES_LIST = ["Asthma", "Bandage", "Blood", "Blood Pressure", "Broke", "Burn", "Cold", "Constipated", "Cut", "Diarrhea", "Disease", "Doctor", "Emergency", "Fever", "Headache", "Infection", "Itch", "Nauseous", "Pain", "Patient", "Redness", "Sore Throat", "Urgent Care (UC)", "Anxiety", "Depressed", "Hospital", "Sprain", "Swallow", "Treatment", "Vomit"]

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100
SEQUENCE_LENGTH = 30

def preprocess_video(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)

    video_reader.release()
    return np.array(frames_list)

def predict_single_action(video_path):
    video_frames = preprocess_video(video_path)
    predictions = model.predict(np.expand_dims(video_frames, axis=0))[0]
    predicted_label = np.argmax(predictions)
    predicted_class_name = CLASSES_LIST[predicted_label]
    # confidence = predictions[predicted_label]
    print("predictions", predicted_class_name)
    return predicted_class_name

@app.post("/predict-video/")
async def predict_video(video_file: UploadFile):
    # Save the video content to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(video_file.file.read())
        temp_file_path = temp_file.name

    predicted_class_name = predict_single_action(temp_file_path)

    # Remove the temporary file
    os.remove(temp_file_path)

    return {"predicted_class": predicted_class_name}
