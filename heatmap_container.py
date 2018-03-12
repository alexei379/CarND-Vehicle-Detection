from collections import deque
import numpy as np
from scipy.ndimage.measurements import label
import cv2
import config


class HeatmapContainer:
    def __init__(self, over_frames=config.Heatmap.NUM_OF_FRAMES_TO_SUM,
                 threshold=config.Heatmap.THRESHOLD, shape=(720, 1280)):
        self.over_frames = over_frames
        self.threshold = threshold
        self.shape = shape

        self.heatmap_queue = deque(maxlen=self.over_frames)
        self.heatmap = np.zeros(shape=shape).astype(int)
        self.thresholded_heatmap = None

        self.labels = []
        self.frame_counter = 0
        self.prev_detections = []

    def add_to_heatmap(self, bbox_list_to_add):
        new_heatmap = np.zeros(shape=self.shape).astype(int)
        for box in bbox_list_to_add:
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            new_heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1 + self.boost(box)
        self.heatmap_queue.append(new_heatmap)
        self.heatmap = sum(self.heatmap_queue)

        self.thresholded_heatmap = np.copy(self.heatmap)
        self.thresholded_heatmap[self.thresholded_heatmap <= self.threshold] = 0
        self.labels = label(self.thresholded_heatmap)

    # reward boxes that land withing previously detected regions
    def boost(self, box):
        margin = config.Heatmap.BOOST_MARGIN
        boost = config.Heatmap.BOOST_AMOUNT

        # car entering from the right
        if self.shape[1] - 180 < box[0][0] and self.shape[0] - 200 > box[0][1]:
            return boost

        for prev_box in self.prev_detections:
            if prev_box[0][0] - margin < box[0][0] and \
                    prev_box[0][1] - margin < box[0][1] and \
                    prev_box[1][0] + margin > box[1][0] and \
                    prev_box[1][1] + margin > box[1][1]:
                return boost

        return 0

    def draw_labeled_bboxes(self, img, color=(0, 0, 255), thickness=2):
        self.frame_counter += 1
        self.prev_detections = []
        # Iterate through all detected cars
        for car_number in range(1, self.labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (self.labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            if abs(bbox[0][0] - bbox[1][0]) * abs(bbox[0][1] - bbox[1][1]) >= config.Heatmap.MIN_AREA_TO_DRAW:
                # save detections
                self.prev_detections.append(bbox)

                # Draw the box on the image
                cv2.rectangle(img, bbox[0], bbox[1], color=color, thickness=thickness)

                # add heatmap min-max
                # sub_heatmap = self.heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
                # cv2.putText(img, str(sub_heatmap.min()) + '-'+str(sub_heatmap.max()), bbox[0], cv2.FONT_HERSHEY_DUPLEX, 0.75, 255)
        # Return the image
        return img

    def render_heatmap(self, thresholded=True):
        if thresholded:
            hm = self.thresholded_heatmap
        else:
            hm = self.heatmap

        boost = 0.1
        return np.dstack((np.clip(hm * boost, 0, 255),
                          np.zeros_like(hm),
                          np.zeros_like(hm))).astype(np.uint8)
