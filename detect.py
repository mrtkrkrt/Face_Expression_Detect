import tensorflow as tf
import cv2 
import numpy as np
import os
from keras.models import model_from_json
from keras.preprocessing import image

classes = os.listdir("train")
model = tf.keras.models.load_model("model.model")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
cap.set(3, 1000)
cap.set(4, 1000)

while True:
    
    ret, frame = cap.read()
    face = face_cascade.detectMultiScale(frame, 1.1, 3)
    
    for (x, y, w, h) in face:
        img = frame[y:y+h, x:x+h,:]
        img = cv2.resize(img, (32,32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = img / 255
        img = tf.expand_dims(img, axis = 0)
        
        result =np.argmax(model.predict(img)[0])
        
        cv2.rectangle(frame, (x,y), (x+h,y+w), (0,0,255))
        cv2.putText(frame, classes[result], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.imshow("happiness Deteector", frame)
        
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()