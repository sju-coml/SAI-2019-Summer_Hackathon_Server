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
        self.face_model.load_model('model_data/yolo_weights.h5')
        self.vidcap = cv2.VideoCapture('data/20190625_134522.mp4')
        self.table = [0, 0, 0, 0]

        if not self.vidcap.isOpened():
            raise IOError("Couldn't open camera")
        
        self.thread = threading.Thread(target=self.detect)
        self.thread.start()

    def detect(self):
        font = cv2.FONT_HERSHEY_SIMPLEX

        while True:
            cnt = []
            tables = []
            for i in range(3):
                all_num = 0
                t_num = [0, 0, 0, 0]
                return_value, frame = self.vidcap.read()
                if not return_value:
                    break

                frame = frame[330:-200, :]
                height, width, _ = frame.shape

                image = Image.fromarray(frame)
                boxes, classes, confidence = self.face_model.detect(image)

                for k, box in enumerate(boxes):
                    (x1, y1, x2, y2) = box
                    if y1 > 10:
                        all_num += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                        cv2.putText(frame, classes[k], (x1, y1), font, 1.5, (0, 255, 255))

                        pos_x = (x1 + x2) / 2
                        pos_y = (y1 + y2) / 2
                        
                        # left part
                        if pos_x < width / 2:
                            # table 2
                            if pos_y > height / 2 - 100:
                                t_num[1] += 1
                            # table 1
                            else:
                                t_num[0] += 1
                        # right part
                        else:
                            # table 4
                            if pos_y > height / 2 - 100:
                                t_num[3] += 1
                            # table 3
                            else:
                                t_num[2] += 1
                    
                cv2.imwrite('img'+str(i)+'.png', frame)
                
                cnt.append(all_num)
                tables.append(t_num)
                time.sleep(2)
            
            self.num_of_people = max(cnt)
            self.num_of_tables = [0, 0, 0, 0]
            for t in range(4):
                self.num_of_tables[t] = max([tables[0][t], tables[1][t], tables[2][t]])

            time.sleep(15)

@app.route("/numofpeople", methods=["GET"])
def get_num_of_people():
    data = {"num": model.num_of_people}
    return flask.jsonify(data)

@app.route("/tablestatus", methods=["GET"])
def get_table_status():
    data = {"t1": model.num_of_tables[0], "t2": model.num_of_tables[1],
            "t3": model.num_of_tables[2], "t4": model.num_of_tables[3]}
    return flask.jsonify(data)
    

if __name__ == "__main__":
    print("Loading keras model")
    global model
    model = DetectModel()
    app.run()
    