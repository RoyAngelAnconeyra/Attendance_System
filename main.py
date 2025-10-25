import cv2 as cv
import numpy as np

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.preprocessing import LabelEncoder

import pickle

import tensorflow as tf
from keras_facenet import FaceNet


facenet = FaceNet()

faces_embeddings = np.load("face-embeddings-done-for-classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)

haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("face-recognition-model.pkl", 'rb'))

cap = cv.VideoCapture(0)

cap.set(cv.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    _, frame = cap.read()
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    for x, y, w, h in faces:
        img = rgb_img[y: y + h, x: x + w]
        img = cv.resize(img, (160,160))
        img = np.expand_dims(img,axis=0)
        ypred = facenet.embeddings(img)
        face_name = model.predict(ypred)
        final_name = encoder.inverse_transform(face_name)[0]
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.putText(frame, str(final_name), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    cv.imshow("Face Recognition:", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows