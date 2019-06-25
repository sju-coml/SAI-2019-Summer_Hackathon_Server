import numpy as np
import flask
import cv2
import os
from PIL import Image
import threading, time
#from detection_model import FaceDetector

app = flask.Flask(__name__)

class DetectModel(object):

    def __init__(self):
        self.num_of_people = 0
        #self.face_model = FaceDetector()
        #self.face_model.load_model('model_data/yolo_face_model.h5')
        self.thread = threading.Thread(target=self.detect)
        self.thread.start()

    def detect(self):
        while True:
            time.sleep(3)
            self.num_of_people += 2

@app.route("/numofpeople", methods=["GET"])
def get_num_of_people():
    data = {"num": model.num_of_people}
    return flask.jsonify(data)

if __name__ == "__main__":
    print("Loading keras model")
    global model
    model = DetectModel()
    app.run()
    