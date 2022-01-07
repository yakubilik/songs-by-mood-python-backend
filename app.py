from flask import Flask, request
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import base64
import io
import cv2
import numpy as np


def img_to_mood(img):
    face_classifier = cv2.CascadeClassifier('haardcascade.xml')
    classifier = load_model('model.h5')

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    labels = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    is_face = False
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        is_face = True
    if is_face:
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
    else:
        label = "No Emotions Detected"

    return label

def read_image(json_data):
    img = json_data["img"]
    img = base64.b64decode(img)
    img = io.BytesIO(img)
    img = Image.open(img)
    cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return cv_image

app = Flask(__name__)
@app.route('/',methods=['GET','POST'])
def home():
    value = request.json
    img = read_image(value)
    result = img_to_mood(img)
    result_json = {"result":result}
    return result_json

if __name__ == '__main__':
    app.run(debug=True)