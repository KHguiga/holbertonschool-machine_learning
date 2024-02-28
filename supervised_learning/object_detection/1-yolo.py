#!/usr/bin/env python3
"""this module contains the class Yolo"""
import tensorflow.keras as K
import numpy as np
import os


def sigmoid(x):
    """sigmoid function"""
    return 1 / (1 + np.exp(-x))


class Yolo:
    """Yolo class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Constructor method
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError("Wrong model file path")

        if not os.path.exists(classes_path):
            raise FileNotFoundError("Wrong classes file path")
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line[:-1] for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process and normalize the output of the YoloV3 model"""
        boxes = []
        box_confidences = []
        box_class_probs = []
        img_h, img_w = image_size
        i = 0
        for output in outputs:
            grid_h, grid_w, nb_box, _ = output.shape
            box_conf = np.expand_dims(sigmoid(output[:, :, :, 4]), axis=-1)
            box_prob = sigmoid(output[:, :, :, 5:])
            box_confidences.append(box_conf)
            box_class_probs.append(box_prob)
            # t_x, t_y : x and y coordinates of the center pt of the anchor box
            # t_w, t_h : width and height of the anchor box
            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]
            # c_x, c_y : is going to represent the grid of indexes
            c_x = np.arange(grid_w)
            c_x = np.tile(c_x, grid_h)
            c_x = c_x.reshape(grid_h, grid_w, 1)

            c_y = np.arange(grid_h)
            c_y = np.tile(c_y, grid_w)
            c_y = c_y.reshape(1, grid_h, grid_w).T

            print(c_x, c_y)

            # p_w, p_h : anchors dimensions in the c

            p_w = self.anchors[i, :, 0]
            p_h = self.anchors[i, :, 1]

            # yolo formula (get the coordinates in the prediction box)
            b_x = (sigmoid(t_x) + c_x)
            b_y = (sigmoid(t_y) + c_y)
            b_w = (np.exp(t_w) * p_w)
            b_h = (np.exp(t_h) * p_h)
            # normalize to the input size
            b_x = b_x / grid_w
            b_y = b_y / grid_h
            b_w = b_w / self.model.input.shape[1].value
            b_h = b_h / self.model.input.shape[2].value
            # scale to the image size (in pixels)
            # top left corner
            x1 = (b_x - b_w / 2) * img_w
            y1 = (b_y - b_h / 2) * img_h
            # bottom right corner
            x2 = (b_x + b_w / 2) * img_w
            y2 = (b_y + b_h / 2) * img_h
            # create the current box
            box = np.zeros((grid_h, grid_w, nb_box, 4))
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2
            boxes.append(box)
            i += 1
        return boxes, box_confidences, box_class_probs