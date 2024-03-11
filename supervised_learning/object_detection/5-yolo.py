#!/usr/bin/env python3

import cv2
import glob
import numpy as np
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

    def process_outputs(self, outputs, image_size):
        """processes the outputs of the model"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            anchors = self.anchors[i]
            g_h, g_w = output.shape[:2]
            
            t_xy = output[..., :2]
            t_wh = output[..., 2:4]
            box_confidence = np.expand_dims(1/(1 + np.exp(-output[..., 4])), axis=-1)
            box_class_prob = 1/(1 + np.exp(-output[..., 5:]))

            b_wh = anchors * np.exp(t_wh)
            b_wh = b_wh / self.model.inputs[0].shape.as_list()[1:3]
            grid = np.tile(np.indices((g_w, g_h)).T, anchors.shape[0]).reshape((g_h, g_w) + anchors.shape)
            b_xy = (1/(1 + np.exp(-t_xy)) + grid) / [g_w, g_h]
            b_xy1 = b_xy - (b_wh / 2)
            b_xy2 = b_xy + (b_wh / 2)
            box = np.concatenate((b_xy1, b_xy2), axis=-1)
            box = box * np.tile(np.flip(image_size, axis=0), 2)
            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """filters all boxes with a score below a specific threshold"""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i, b in enumerate(boxes):
            bc = box_confidences[i]
            bcp = box_class_probs[i]

            bs = bc * bcp
            bcl = np.argmax(bs, axis=-1)
            bcs = np.max(bs, axis=-1)
            idx = np.where(bcs >= self.class_t)

            filtered_boxes.append(b[idx])
            box_classes.append(bcl[idx])
            box_scores.append(bcs[idx])

        return np.concatenate(filtered_boxes), np.concatenate(box_classes), np.concatenate(box_scores)

    def non_max_suppression(self, boxes, box_classes, box_scores):
        """performs non max suppression on all remaining boxes"""
        filtered_boxes = []
        filtered_classes = []
        filtered_scores = []
        
        classes = np.unique(box_classes)
        # perform non max suppression on each class
        for c in classes:
            # get the box info for all boxes in this class
            idx = np.where(box_classes == c)
            b = boxes[idx]
            bcl = box_classes[idx]
            bs = box_scores[idx]

            # get the indices of boxes ordered by score
            ordered_idx = np.flip(bs.argsort(), axis=0)
            # indices of boxes to keep
            keep_idx = []
            # calculate IOU relative to the max and repeat
            # as long as there are boxes remaining
            while ordered_idx.size > 1:
                maximum, others = ordered_idx[0], ordered_idx[1:]
                keep_idx.append(maximum)

                # get coordinates for intersection
                x1 = np.maximum(b[maximum][0], b[others][:, 0])
                y1 = np.maximum(b[maximum][1], b[others][:, 1])
                x2 = np.minimum(b[maximum][2], b[others][:, 2])
                y2 = np.minimum(b[maximum][3], b[others][:, 3])

                intersection = (x2 - x1) * (y2 - y1)

                a_max = (b[maximum][2] - b[maximum][0]) * (b[maximum][3] - b[maximum][1])
                a_others = (b[others][:, 2] - b[others][:, 0]) * (b[others][:, 3] - b[others][:, 1])

                union = a_max + a_others - intersection

                IOU = intersection / union

                # indices that are below the threshold 
                below_idx = np.where(IOU < self.nms_t)[0]
                ordered_idx = ordered_idx[below_idx + 1]

            if ordered_idx.size > 0:
                keep_idx.append(ordered_idx[0])
            keep_idx = np.array(keep_idx)

            filtered_boxes.append(b[keep_idx])
            filtered_classes.append(bcl[keep_idx])
            filtered_scores.append(bs[keep_idx])

        return np.concatenate(filtered_boxes), np.concatenate(filtered_classes), np.concatenate(filtered_scores)

    @staticmethod
    def load_images(folder_path):
        """load all images in a folder"""
        images = []
        image_paths = []
        for file_path in glob.glob(folder_path + '/*.*'):
            image = cv2.imread(file_path)
            if image is not None:
                images.append(image)
                image_paths.append(file_path)
        return images, image_paths
    
    def preprocess_images(self, images):
        """preprocess all images as inputs for the Darknet model"""
        pimages = []
        image_shapes = []
        target_shape = tuple(self.model.inputs[0].shape.as_list()[1:3])
        for image in images:
            pimage = cv2.resize(image, target_shape, interpolation=cv2.INTER_CUBIC) / 255
            pimages.append(pimage)
            image_shapes.append(image.shape[:2])
        return np.array(pimages), np.array(image_shapes)