import cv2

import config
import image_features
import numpy as np
from sklearn.externals import joblib

import visualization


def load(filename):
    return joblib.load(filename)


def save(clf, filename):
    joblib.dump(clf, filename)


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image and resize to the size of training data
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        features = image_features.single_image_features(test_img, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_f=spatial_feat,
                            hist_f=hist_feat, hog_f=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, vis=False, min_confidence=0.2):
    on_windows = []
    on_confidence = []

    img_tosearch = img[ystart:ystop, xstart:xstop, :]

    ctrans_tosearch = image_features.convert_to_color_space(img_tosearch, config.Classifier.COLOR_SPACE)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    if vis:
        # Compute individual channel HOG features for the entire image
        hog1, hog1_viz = image_features.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False, vis=vis)
        hog2, hog2_viz = image_features.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False, vis=vis)
        hog3, hog3_viz = image_features.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False, vis=vis)
    else:
        # Compute individual channel HOG features for the entire image
        hog1 = image_features.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = image_features.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = image_features.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = image_features.bin_spatial(subimg, size=spatial_size)
            hist_features = image_features.color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                confidence = svc.decision_function(test_features)[0]
                if confidence < min_confidence:
                    continue
                on_confidence.append(confidence)

                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)

                on_windows.append(((xbox_left + xstart, ytop_draw + ystart),
                              (xbox_left + win_draw + xstart, ytop_draw + win_draw + ystart)))
    # print(on_confidence)
    if vis:
        hog_combined_img = np.dstack((hog1_viz, hog2_viz, hog3_viz)) * 255
        return on_windows, on_confidence, hog_combined_img.clip(0, 255).astype(np.uint8)
        #return on_windows, visualization.draw_boxes(img, on_windows, (255, 0, 0), 3)
    else:
        return on_windows, on_confidence