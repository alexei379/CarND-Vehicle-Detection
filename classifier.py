import cv2

import config
import image_features
import numpy as np


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, xstart, xstop, scale,
              cls, X_scaler,
              orient, pix_per_cell, cell_per_block,
              spatial_size, hist_bins,
              vis=False, min_confidence=0.2, cells_per_step=1):

    on_windows = []

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

    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    if vis:
        # Compute individual channel HOG features for the entire image + visualization
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
            test_prediction = cls.predict(test_features)

            if test_prediction == 1:
                confidence = cls.decision_function(test_features)[0]
                if confidence < min_confidence:
                    continue

                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)

                on_windows.append(((xbox_left + xstart, ytop_draw + ystart),
                              (xbox_left + win_draw + xstart, ytop_draw + win_draw + ystart)))
    if vis:
        hog_combined_img = np.dstack((hog1_viz, hog2_viz, hog3_viz)) * 255
        return on_windows, hog_combined_img.clip(0, 255).astype(np.uint8)
    else:
        return on_windows