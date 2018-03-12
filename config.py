class Classifier:
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

    CLS_FILE = 'cls_' + COLOR_SPACE + '.pkl'
    SCALER_FILE = 'scaler_' + COLOR_SPACE + '.pkl'


class Trainer:
    TRAIN_ON_SUBSET = -1

class Pipeline:
    DEBUG = True
    IMG_INPUT = "test_images/*.jpg"
    IMG_OUTPUT_DIR = "output_images/"
    VIDEO_INPUT = "test_video.mp4"
    VIDEO_OUTPUT = "output_video/" + VIDEO_INPUT
    SCALES = [
        # (scale, ystart, ystop, xstart, xstop)
        (0.5, 412, 466, 620, 1020),
        (1, 400, 496, 600, 1272),
        (1.25, 368, 528, 600, 1280),
        (1.5, 368, 560, 600, 1280),
        (2, 368, 624, 600, 1280),
    ]
    MIN_CONFIDENCE = 0.4

class Heatmap:
    BOOST_MARGIN = 25
    BOOST_AMOUNT = 10
    MIN_AREA_TO_DRAW = 512
    NUM_OF_FRAMES_TO_SUM = 12
    THRESHOLD = 200

class Dataset:
    DEFAULT_BASE_DIR = "object_dataset/default-project-dataset/"
    DEFAULT_VEHICLES = DEFAULT_BASE_DIR + "vehicles/**/*.png"
    DEFAULT_NON_VEHICLES = DEFAULT_BASE_DIR + "non-vehicles/**/*.png"

