from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import face_recognition
import keras
from keras.models import load_model
import cv2

emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}
model = load_model("model_v6_23.hdf5")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    gray_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    face_locations = face_recognition.face_locations(gray_small_frame)
    face_images = []
    for face_points in face_locations:
        top, right, bottom, left = face_points
        face_image = gray_small_frame[top:bottom, left:right]
        face_image = cv2.resize(face_image, (48,48))
        face_image = np.reshape(face_image, [48, 48, 1])
        face_images.append(face_image)
    face_images = np.asarray(face_images)
    if face_images.shape[0]:
        preds = np.argmax(model.predict(face_images))
        label_map = dict((v,k) for k,v in emotion_dict.items())
        #preds = preds.tolist()
        #predicted_labels = [label_map[p] for p in preds]
        #print(predicted_labels)
        print(label_map[preds])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
