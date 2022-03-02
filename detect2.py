import tensorflow as tf
import cv2 
import numpy as np
import os
from keras.models import model_from_json
from keras.preprocessing import image
from PIL import Image

classes = os.listdir("train")
model = tf.keras.models.load_model("model.model")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = Image("ahmet.jpg")

face = face_cascade.detectMultiScale(img, 1.1, 3)
