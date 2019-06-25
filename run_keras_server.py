import numpy as np
import flask
import cv2
import os
from PIL import Image
import threading, time
from detection_model import FaceDetector
from flask_cors import CORS

app = flask.Flask(__name__)
CORS(app)

class DetectModel(object):

    def __init__(self):
        self.num_of_people = 0
        self.face_model = FaceDetector()
        self.face_model.load_model('D:/python_projects/seat_check_rest_api/model_data/yolo_weights.h5')
        self.vidcap = cv2.VideoCapture('D:/python_projects/seat_check_rest_api/data/20190625_134522.mp4')
        self.table = [0, 0, 0, 0]

        if not self.vidcap.isOpened():
            raise IOError("Couldn't open camera")
        
        self.thread = threading.Thread(target=self.detect)
        self.thread.start()

    def detect(self):
        font = cv2.FONT_HERSHEY_SIMPLEX

        """
            status 0 : no person,
            status 1 : There are people,
            status 2 : vacuum,
        """
        table_status = []
        for t_number in range(4):
            table_status.append({
                "status": 0,
                "detected_num": 0,
                "time": 60,
                "count": 0
            })

        while True:
            # The number of all people
            all_num = 0
            # The number of people in each table
            t_num = [0, 0, 0, 0]

            return_value, frame = self.vidcap.read()
            if not return_value:
                break

            # detect
            frame = frame[330:-200, :]
            height, width, _ = frame.shape
            image = Image.fromarray(frame)
            boxes, classes, confidence = self.face_model.detect(image)

            # get table infor
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
                
            cv2.imwrite('img.png', frame)
            self.num_of_people = all_num
            self.num_of_tables = t_num

            for t, status in enumerate(table_status):
                table_status[t]['time'] -= 1
                if self.num_of_tables[t] > 0:
                    table_status[t]['count'] += 1
                
                if status['status'] == 0 and status['time'] == 0:
                    if status['count'] >= 40:
                        table_status[t]['status'] = 1
                        table_status[t]['time'] = 60
                        table_status[t]['count'] = 0
                    else:
                        table_status[t]['time'] = 60
                        table_status[t]['count'] = 0
                else if status['status'] == 1 and status['time'] == 0:
                    if status['count'] < 10:
                        table_status[t]['status'] = 2
                        table_status[t]['time'] = 540
                        table_status[t]['count'] = 0
                    else:
                        table_status[t]['time'] = 60
                        table_status[t]['count'] = 0
                else if status['status'] == 2:
                    pass
            
            time.sleep(1)            


@app.route("/numofpeople", methods=["POST"])
def get_num_of_people():
    data = {"num": model.num_of_people}
    return flask.jsonify(data)

@app.route("/tablestatus", methods=["POST"])
def get_table_status():
    data = {"t1": model.num_of_tables[0], "t2": model.num_of_tables[1],
            "t3": model.num_of_tables[2], "t4": model.num_of_tables[3]}
    return flask.jsonify(data)


global model
model = DetectModel()
app.run(host='0.0.0.0', port=5000)
    

"""
if __name__ == "__main__":
    print("Loading keras model")
    global model
    model = DetectModel()
    app.run()
"""