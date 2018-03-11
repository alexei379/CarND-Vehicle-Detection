import cv2
import numpy as np

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imgcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imgcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imgcopy


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 3)
    # Return the image
    return img


def save_image(img, filename, colorspace=cv2.COLOR_RGB2BGR):
    if len(img.shape) > 2:
        cv2.imwrite(filename, cv2.cvtColor(img, colorspace))
    else:
        img = np.dstack((img, img, img)) * 255
        cv2.imwrite(filename, img)


def load_image(file_name, colorspace=cv2.COLOR_BGR2RGB):
    return cv2.cvtColor(cv2.imread(file_name), colorspace)

