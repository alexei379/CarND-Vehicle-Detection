import config
import datasets
import image_features
import numpy as np
import visualization
import time
import classifier
import glob
import cv2

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ds = datasets.DefaultDatasetLoader(config.Pipeline.DEBUG)

# save some test images
for i in range(0, 10):
    color_space = 'RGB'
    orient = 6
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 0
    spatial_size = (16, 16)
    hist_bins = 16
    spatial_f = True
    hist_f = True
    hog_f = True

    car_image_filename = ds.vehicles[np.random.randint(0, len(ds.vehicles))]
    car_image = visualization.load_image(car_image_filename)

    noncar_image_filename = ds.non_vehicles[np.random.randint(0, len(ds.non_vehicles))]
    noncar_image = visualization.load_image(noncar_image_filename)

    car_features, car_hog_image = image_features.single_image_features(car_image,
                                                                       color_space,
                                                                       spatial_size,
                                                                       hist_bins,
                                                                       orient,
                                                                       pix_per_cell,
                                                                       cell_per_block,
                                                                       hog_channel,
                                                                       spatial_f,
                                                                       hist_f,
                                                                       hog_f,
                                                                       True)

    noncar_features, noncar_hog_image = image_features.single_image_features(noncar_image,
                                                                             color_space,
                                                                             spatial_size,
                                                                             hist_bins,
                                                                             orient,
                                                                             pix_per_cell,
                                                                             cell_per_block,
                                                                             hog_channel,
                                                                             spatial_f,
                                                                             hist_f,
                                                                             hog_f,
                                                                             True)
    img_prefix = config.Pipeline.IMG_OUTPUT_DIR + "hod_demo/" + str(i)
    visualization.save_image(car_image, img_prefix + "_1_car.png")
    visualization.save_image(car_hog_image, img_prefix + "_2_car_hog.png")
    visualization.save_image(noncar_image, img_prefix + "_3_noncar.png")
    visualization.save_image(noncar_hog_image, img_prefix + "_4_noncar_hog.png")

X_scaler = None
svc = None

if config.Classifier.RECALCULATE:
    color_space = config.Classifier.COLOR_SPACE
    orient = config.Classifier.ORIENT
    pix_per_cell = config.Classifier.PIX_PER_CELL
    cell_per_block = config.Classifier.CELL_PER_BLOCK
    hog_channel = config.Classifier.HOG_CHANNEL
    spatial_size = config.Classifier.SPATIAL_SIZE
    hist_bins = config.Classifier.HIST_BINS
    spatial_f = config.Classifier.SPATIAL_F
    hist_f = config.Classifier.HIST_F
    hog_f = config.Classifier.HOG_F

    t = time.time()

    n_samples = config.Classifier.TRAIN_ON_SUBSET
    test_cars = None
    test_noncars = None

    if n_samples > 0:
        random_idxs = np.random.randint(0, len(ds.vehicles), n_samples)
        test_cars = np.array(ds.vehicles)[random_idxs]
        test_noncars = np.array(ds.non_vehicles)[random_idxs]
    else:
        test_cars = ds.vehicles
        test_noncars = ds.non_vehicles

    car_features = image_features.extract_features(test_cars,
                                                   color_space,
                                                   spatial_size,
                                                   hist_bins,
                                                   orient,
                                                   pix_per_cell,
                                                   cell_per_block,
                                                   hog_channel,
                                                   spatial_f,
                                                   hist_f,
                                                   hog_f)

    noncar_features = image_features.extract_features(test_noncars,
                                                      color_space,
                                                      spatial_size,
                                                      hist_bins,
                                                      orient,
                                                      pix_per_cell,
                                                      cell_per_block,
                                                      hog_channel,
                                                      spatial_f,
                                                      hist_f,
                                                      hog_f)

    print(time.time() - t, 'seconds to compute features')

    X = np.vstack((car_features, noncar_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    # labels
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1, random_state=rand_state)

    print('Feature vector length', len(X_train[0]))

    svc = LinearSVC()
    t = time.time()
    svc.fit(X_train, y_train)

    print(time.time() - t, 'Seconds to train')
    print('test accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    classifier.save(X_scaler, config.Classifier.SCALER_FILE)
    classifier.save(svc, config.Classifier.CLS_FILE)
else:
    X_scaler = classifier.load(config.Classifier.SCALER_FILE)
    svc = classifier.load(config.Classifier.CLS_FILE)

y_start_stop = [400, 656]
overlap = 0.5

color_space = config.Classifier.COLOR_SPACE
orient = config.Classifier.ORIENT
pix_per_cell = config.Classifier.PIX_PER_CELL
cell_per_block = config.Classifier.CELL_PER_BLOCK
hog_channel = config.Classifier.HOG_CHANNEL
spatial_size = config.Classifier.SPATIAL_SIZE
hist_bins = config.Classifier.HIST_BINS
spatial_f = config.Classifier.SPATIAL_F
hist_f = config.Classifier.HIST_F
hog_f = config.Classifier.HOG_F

for img_src in glob.glob(config.Pipeline.IMG_INPUT):
    t1 = time.time()
    img = visualization.load_image(img_src)
    draw_img = np.copy(img)
    print(np.min(img), np.max(img))

    windows = image_features.slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop,
                                          xy_window=(96, 96), xy_overlap=(overlap, overlap))

    hot_windows = classifier.search_windows(img, windows, svc, X_scaler, color_space=color_space,
                                            spatial_size=spatial_size, hist_bins=hist_bins,
                                            orient=orient, pix_per_cell=pix_per_cell,
                                            cell_per_block=cell_per_block,
                                            hog_channel=hog_channel, spatial_feat=spatial_f,
                                            hist_feat=hist_f, hog_feat=hog_f)

    window_img = visualization.draw_boxes(draw_img, hot_windows, color=(0, 0, 255), thick=6)
    print(time.time() - t1, 'Seconds to process image searching', len(windows), 'windows')
    visualization.save_image(window_img, config.Pipeline.IMG_OUTPUT_DIR + img_src)

