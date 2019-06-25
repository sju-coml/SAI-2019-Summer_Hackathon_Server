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
        self.model_image_size = (480,480)
        self.score = score
        self.iou = iou
        self.input_image_shape = K.placeholder(shape=(2,))
        self.gpu_num = gpu_num
    
    def generate(self, model_path, anchors, score):
        print(model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        num_anchors = len(anchors)
        num_classes = 1

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
        self.yolo_model = self.generate(yolo_model_path, self.anchors, 0.2)


    def detect(self, frame):
        faces = []
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

        # face detection
        confidences = []
        for i, score in enumerate(scores):
            box = boxes[i]
            y1, x1, y2, x2 = box
            y1 = max(0, np.floor(y1 + 2.5).astype('int32'))
            x1 = max(0, np.floor(x1 + 2.5).astype('int32'))
            y2 = min(frame.size[1], np.floor(y2 + 2.5).astype('int32'))
            x2 = min(frame.size[0], np.floor(x2 + 2.5).astype('int32'))
            confidences.append(score)
            faces.append((x1, y1, x2, y2))
        
        return faces, confidences


    def close_session(self):
        self.sess.close()