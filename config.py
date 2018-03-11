class Classifier:
    RECALCULATE = True
    COLOR_SPACE = 'YCrCb'
    ORIENT = 8
    PIX_PER_CELL = 8
    CELL_PER_BLOCK = 2
    HOG_CHANNEL = 'ALL'
    SPATIAL_SIZE = (16, 16)
    HIST_BINS = 64
    SPATIAL_F = True
    HIST_F = True
    HOG_F = True
    TRAIN_ON_SUBSET = -1

    CLS_FILE = 'cls_' + COLOR_SPACE + '.pkl'
    SCALER_FILE = 'scaler_' + COLOR_SPACE + '.pkl'

class Trainer:
    None


class Pipeline:
    DEBUG = True
    IMG_INPUT = "test_images/*.jpg"
    IMG_OUTPUT_DIR = "output_images/"
    VIDEO_INPUT = "test_video.mp4"
    VIDEO_OUTPUT = "output_video/" + VIDEO_INPUT

class Dataset:
    DEFAULT_BASE_DIR = "object_dataset/default-project-dataset/"
    DEFAULT_VEHICLES = DEFAULT_BASE_DIR + "vehicles/**/*.png"
    DEFAULT_NON_VEHICLES = DEFAULT_BASE_DIR + "non-vehicles/**/*.png"
