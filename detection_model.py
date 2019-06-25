import os
import numpy as np
import cv2
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from yolo3.model import yolo_eval, yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model

class FaceDetector(object):
    def __init__(self, score=0.3, iou=0.45, gpu_num=1):
        self.sess = K.get_session()
        self.anchors = np.array([10.0,13.0, 16.0,30.0, 33.0,23.0, 30.0,61.0, 62.0,45.0, 59.0,119.0, 116.0,90.0, 156.0,198.0, 373.0,326.0]).reshape(-1, 2)
        self.model_image_size = (416,416)
        self.score = score
        self.iou = iou
        self.input_image_shape = K.placeholder(shape=(2,))
        self.gpu_num = gpu_num
    
    def generate(self, model_path, anchors, score):
        print(model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        num_anchors = len(anchors)
        self.coco_classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus",
                   "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                   "parking meter", "bench", "bird", "cat", "dog", "horse",
                   "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                   "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                   "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                   "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                   "fork", "knife", "spoon", "bowl", "banana", "apple",
                   "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
                   "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                   "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
                   "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                   "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

        #num_classes = 1
        num_classes = len(self.coco_classes)

        model = None
        try:
            model = load_model(model_path, compile=False)
        except:
            model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            model.load_weights(model_path)
        
        
        if self.gpu_num >=2:
            model = multi_gpu_model(model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(model.output, anchors,
                num_classes, self.input_image_shape,
                score_threshold=score, iou_threshold=self.iou)

        return {'model':model, 'boxes':boxes, 'scores': scores, 'classes': classes}


    def load_model(self, yolo_model_path):
        self.yolo_model = self.generate(yolo_model_path, self.anchors, 0.3)


    def detect(self, frame):
        objs = []
        class_names = []
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'

        new_image_size = (frame.width - (frame.width % 32),
                          frame.height - (frame.height % 32))
        boxed_frame = letterbox_image(frame, new_image_size)
        image_data = np.array(boxed_frame, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        boxes, scores, classes = self.sess.run(
            [self.yolo_model['boxes'], self.yolo_model['scores'], self.yolo_model['classes']],
            feed_dict={
                self.yolo_model['model'].input: image_data,
                self.input_image_shape: [frame.size[1], frame.size[0]],
                #K.learning_phase(): 0
            })

        # detection
        confidences = []
        for i, score in enumerate(scores):
            box = boxes[i]
            class_name = self.coco_classes[classes[i]]
            if class_name == 'person':
                y1, x1, y2, x2 = box
                y1 = max(0, np.floor(y1 + 2.5).astype('int32'))
                x1 = max(0, np.floor(x1 + 2.5).astype('int32'))
                y2 = min(frame.size[1], np.floor(y2 + 2.5).astype('int32'))
                x2 = min(frame.size[0], np.floor(x2 + 2.5).astype('int32'))
                confidences.append(score)
                objs.append((x1, y1, x2, y2))
                class_names.append(class_name)
        
        return objs, class_names, confidences


    def close_session(self):
        self.sess.close()