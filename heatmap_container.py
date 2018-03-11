from collections import deque
import numpy as np
from scipy.ndimage.measurements import label
import cv2

class HeatmapContainer:
    def __init__(self, over_frames=10, threshold=5, shape=(720, 1280)):
        self.over_frames = over_frames
        self.threshold = threshold
        self.shape = shape

        self.heatmap_queue = deque(maxlen=self.over_frames)
        self.confidence_queue = deque(maxlen=self.over_frames)
        self.heatmap = np.zeros(shape=shape).astype(int)
        self.confidence_heatmap = np.zeros(shape=shape).astype(float)
        self.thresholded_heatmap = None
        self.labels = []
        self.frame_counter = 0


    def add_to_heatmap(self, bbox_list_to_add, confidence):
        if len(self.heatmap_queue) < self.over_frames:
            self.heatmap_queue.append(bbox_list_to_add)
            self.confidence_queue.append(confidence)
            self.add_bb_list(bbox_list_to_add, confidence, coeff=1)
        else:
            bbox_list_to_remove = self.heatmap_queue.popleft()
            confidence_to_remove = self.confidence_queue.popleft()
            self.add_bb_list(bbox_list_to_remove, confidence_to_remove, coeff=-1)
            self.heatmap_queue.append(bbox_list_to_add)
            self.confidence_queue.append(confidence)
            self.add_bb_list(bbox_list_to_add, confidence, coeff=1)
        '''
        if self.heatmap_queue is not None:
            bbox_list_to_remove = self.heatmap_queue.popleft()
            self.add_bb_list(bbox_list_to_remove, coeff=-1)
            self.heatmap_queue.append(bbox_list_to_add)
            self.add_bb_list(bbox_list_to_add, coeff=1)
        else:
            self.heatmap_queue = deque([bbox_list_to_add for i in range(0, self.over_frames)], maxlen=self.over_frames)
            self.add_bb_list(bbox_list_to_add, coeff=self.over_frames)
        '''
        self.thresholded_heatmap = np.copy(self.heatmap)
        self.thresholded_heatmap[self.thresholded_heatmap <= self.threshold] = 0
        self.labels = label(self.thresholded_heatmap)


    def add_bb_list(self, bbox_list, confidence, coeff=1):
        # Iterate through list of bboxes
        for idx, box in enumerate(bbox_list):
            # Add += coef for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += coeff
            self.confidence_heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += coeff * confidence[idx]

    def draw_labeled_bboxes(self, img, color=(0, 0, 255), thickness=4):
        self.frame_counter += 1
        # Iterate through all detected cars
        for car_number in range(1, self.labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (self.labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            if abs(bbox[0][0] - bbox[1][0]) * abs(bbox[0][1] - bbox[1][1]) >= 1500:
                # Draw the box on the image
                cv2.rectangle(img, bbox[0], bbox[1], color=color, thickness=thickness)

                # add heatmap min-max
                sub_heatmap = self.heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
                cv2.putText(img, str(sub_heatmap.min()) + '-'+str(sub_heatmap.max()), bbox[0], cv2.FONT_HERSHEY_DUPLEX, 0.75, 255)

                # add confidence avg
                sub_confidence = self.confidence_heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
                cv2.putText(img, str(sub_confidence.mean()), (bbox[0][0], bbox[1][1] + 25), cv2.FONT_HERSHEY_DUPLEX, 0.75, 255)
        # Return the image
        return img


    def render_heatmap(self, thresholded=True):
        if thresholded:
            hm = self.thresholded_heatmap
        else:
            hm = self.heatmap

        boost = 16
        return np.dstack((np.clip(hm * boost, 0, 255),  # R
                                     np.zeros_like(hm),  # G
                                     np.zeros_like(hm))).astype(np.uint8)  # B
