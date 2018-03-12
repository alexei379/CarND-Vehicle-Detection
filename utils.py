from sklearn.externals import joblib
import cv2
import numpy as np

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imgcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imgcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imgcopy

def save_image(img, filename, colorspace=cv2.COLOR_RGB2BGR):
    if len(img.shape) > 2:
        cv2.imwrite(filename, cv2.cvtColor(img, colorspace))
    else:
        img = np.dstack((img, img, img)) * 255
        cv2.imwrite(filename, img)


def load_image(file_name, colorspace=cv2.COLOR_BGR2RGB):
    return cv2.cvtColor(cv2.imread(file_name), colorspace)


def load_classifier(filename):
    return joblib.load(filename)


def save_classifier(clf, filename):
    joblib.dump(clf, filename)