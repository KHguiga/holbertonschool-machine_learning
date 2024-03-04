#!/usr/bin/env python3

import tensorflow.keras as K

class Yolo:
    """Class to perform the Yolo algorithm on image data"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """initializes the Yolo class"""
        self.class_t = class_t # class score threshold
        self.nms_t = nms_t # non max suppression threshold
        self.model = K.models.load_model(model_path) # keras darknet model
        self.anchors = anchors # anchor boxes
        with open(classes_path) as f:
            self.class_names = [class_name.strip() for class_name in f.readlines()]
