import numpy as np
import flask
import cv2
import os
from PIL import Image
import threading, time
from detection_model import FaceDetector

app = flask.Flask(__name__)

class DetectModel(object):

    def __init__(self):
        self.num_of_people = 0
        self.face_model = FaceDetector()
        self.face_model.load_model('model_data/yolo_face_model.h5')
        self.vidcap = cv2.VideoCapture(0)

        if not self.vidcap.isOpened():
            raise IOError("Couldn't open camera")
        
        self.thread = threading.Thread(target=self.detect)
        self.thread.start()

    def detect(self):
        while True:
            cnt = []
            for i in range(3):
                return_value, frame = self.vidcap.read()
                if not return_value:
                    break
                image = Image.fromarray(frame)
                boxes, confidence = self.face_model.detect(image)
                print(boxes)
                cnt.append(len(boxes))
            
            self.num_of_people = max(cnt)

            time.sleep(60)

@app.route("/numofpeople", methods=["GET"])
def get_num_of_people():
    data = {"num": model.num_of_people}
    return flask.jsonify(data)

if __name__ == "__main__":
    print("Loading keras model")
    global model
    model = DetectModel()
    app.run()
    