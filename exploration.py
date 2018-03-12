import config
import datasets
import image_features
import numpy as np
import utils
import classifier
import glob
import cv2
import os


import trainer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

'''
cls = SVC(kernel='rbf', C=10, gamma=0.0001)
trn = trainer.Trainer()
trn.extract_training_data()
trn.train(cls=cls)
'''
'''
cls = LinearSVC()
trn = trainer.Trainer()
trn.extract_training_data()
trn.train(cls=cls)
'''

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

scales = config.Pipeline.SCALES

# save some test images
ds = datasets.DefaultDatasetLoader(config.Pipeline.DEBUG)

for i in range(0, 10):
    img_prefix = config.Pipeline.IMG_OUTPUT_DIR + "hod_demo/" + str(i)

    car_image_filename = ds.vehicles[np.random.randint(0, len(ds.vehicles))]
    car_image = utils.load_image(car_image_filename)

    noncar_image_filename = ds.non_vehicles[np.random.randint(0, len(ds.non_vehicles))]
    noncar_image = utils.load_image(noncar_image_filename)

    utils.save_image(car_image, img_prefix + "_1_car.png")
    utils.save_image(noncar_image, img_prefix + "_3_noncar.png")

    for c in [0, 1, 2]:
        for o in [6, 8, 12]:

            car_features, car_hog_image = image_features.single_image_features(car_image,
                                                                               color_space,
                                                                               spatial_size,
                                                                               hist_bins,
                                                                               o,
                                                                               pix_per_cell,
                                                                               cell_per_block,
                                                                               c,
                                                                               spatial_f,
                                                                               hist_f,
                                                                               hog_f,
                                                                               True)

            noncar_features, noncar_hog_image = image_features.single_image_features(noncar_image,
                                                                                     color_space,
                                                                                     spatial_size,
                                                                                     hist_bins,
                                                                                     o,
                                                                                     pix_per_cell,
                                                                                     cell_per_block,
                                                                                     c,
                                                                                     spatial_f,
                                                                                     hist_f,
                                                                                     hog_f,
                                                                                     True)

            utils.save_image(car_hog_image, img_prefix + "_2_car_hog" + "_c" + str(c)+ "_o" + str(o) + ".png")
            utils.save_image(noncar_hog_image, img_prefix + "_3_noncar_hog" + "_c" + str(c)+ "_o" + str(o) + ".png")



X_scaler = utils.load_classifier(config.Classifier.SCALER_FILE)
svc = utils.load_classifier(config.Classifier.CLS_FILE)

for img_src in glob.glob(config.Pipeline.IMG_INPUT):
    hog_vis = True
    img = utils.load_image(img_src)
    img_filename = os.path.basename(img_src)

    windows = []
    print(img_filename)
    for s in scales:
        scale, ystart, ystop, xstart, xstop = s
        px_scale = int(scale * 64)

        all_boxes = image_features.slide_window(img, [xstart, xstop], [ystart, ystop], (px_scale, px_scale), (0.875, 0.875))

        windows_s, hog_img = classifier.find_cars(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient, pix_per_cell,
                                         cell_per_block, spatial_size, hist_bins, hog_vis)

        hog_combined = np.copy(img)
        hog_img = cv2.resize(hog_img, (0,0), fx=scale, fy=scale)
        hog_combined[ystart:ystart+hog_img.shape[0], xstart:xstart + hog_img.shape[1]] = hog_img


        out_img_s = utils.draw_boxes(img, all_boxes, (255, 0, 0), 1)
        out_img_s = utils.draw_boxes(out_img_s, windows_s, (0, 255, 0), 1)
        hog_combined = utils.draw_boxes(hog_combined, windows_s, (0, 255, 0), 1)

        utils.save_image(hog_combined, config.Pipeline.IMG_OUTPUT_DIR + img_filename +'_' + str(px_scale) + '_hog.jpg')
        utils.save_image(out_img_s, config.Pipeline.IMG_OUTPUT_DIR + img_filename + '_' + str(px_scale) + '_boxes.jpg')

        windows.extend(windows_s)


    out_img = utils.draw_boxes(img, windows, (0, 255, 255), 1)
    utils.save_image(out_img, config.Pipeline.IMG_OUTPUT_DIR + img_filename + '_all_detected_boxes.jpg')
