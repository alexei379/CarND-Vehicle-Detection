from collections import deque
import numpy as np
from scipy.ndimage.measurements import label
import cv2

class HeatmapContainer:
    def __init__(self, over_frames=10, threshold=5, shape=(1280, 720)):
        self.over_frames = over_frames
        self.threshold = threshold
        self.heatmap_queue = None
        self.heatmap = np.zeros(shape=shape)
        self.thresholded_heatmap = None
        self.labels = []


    def add_heatmap(self, bbox_list_to_add):
        if self.heatmap_queue is not None:
            bbox_list_to_remove = self.heatmap_queue.popleft()
            self.add_bb_list(bbox_list_to_remove, -1)
            self.heatmap_queue.append(bbox_list_to_add)
            self.add_bb_list(bbox_list_to_add, 1)
        else:
            self.heatmap_queue = deque([bbox_list_to_add for i in range(0, self.over_frames)], maxlen=self.over_frames)
            self.add_bb_list(bbox_list_to_add, maxlen=self.over_frames)

        self.thresholded_heatmap = np.copy(self.heatmap)
        self.thresholded_heatmap[self.thresholded_heatmap <= self.threshold] = 0

    def add_bb_list(self, bbox_list, coef=1):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += coef for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += coef

    def draw_labeled_bboxes(self, img, color=(0, 0, 255), thickness=4):
        # Iterate through all detected cars
        for car_number in range(1, self.labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (self.labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], color=color, thickness=thickness)
        # Return the image
        return img


    def render_heatmap(self):
        None