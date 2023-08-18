import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit_webrtc
import os
import random
import pygame

# Modeli yükleyin
model = load_model("animal_classifier.h5")
class_names = ['bird', 'butterfly', 'cat', 'cow', 'dog', 'elephant', 'frog', 'hen', 'horse', 'lion', 'sheep',
               'squirrel']

pygame.mixer.init()


def classify_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    resized_frame = tf.image.convert_image_dtype(resized_frame, tf.float32)
    input_data = tf.expand_dims(resized_frame, 0)

    predictions = model.predict(input_data)
    max_probability = tf.reduce_max(predictions).numpy()

    if max_probability < 0.7:
        return "Not Classified..."
    else:
        predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
        return class_names[predicted_class]


def play_random_sound_from_folder(predicted_class, last_class_played):
    folder_path = f"{predicted_class}/"
    sound_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

    if not sound_files:
        return

    if predicted_class != last_class_played:  # Eğer sınıf değiştiyse
        pygame.mixer.stop()  # Eğer çalıyorsa mevcut sesi durdur

        random_sound_file = random.choice(sound_files)
        sound_path = os.path.join(folder_path, random_sound_file)

        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()

    return predicted_class  # Şimdi çalınan sınıfı geri döndürün


class VideoTransformer(streamlit_webrtc.VideoTransformerBase):
    def __init__(self):
        self.last_class_played = None  # Son çalınan sınıfı tutmak için

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        predicted_class = classify_frame(img)

        # Eğer bir hayvan sınıfı tespit edilirse sesini çalma
        if predicted_class != "Not Classified...":
            self.last_class_played = play_random_sound_from_folder(predicted_class, self.last_class_played)

        label = f"Predicted Class: {predicted_class}"
        cv2.putText(img, label, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img


def app():
    st.title("Animal Classifier")
    st.write("This app classifies the animal you show in the webcam feed!")

    webrtc_ctx = streamlit_webrtc.webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)


if __name__ == "__main__":
    app()
