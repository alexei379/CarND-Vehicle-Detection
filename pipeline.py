import config
import datasets
import image_features
import numpy as np
import visualization
import time
import classifier
import glob
import cv2
from scipy.ndimage.measurements import label
import os
import trainer
from sklearn.svm import SVC

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from heatmap_container import HeatmapContainer


# ds = datasets.DefaultDatasetLoader(config.Pipeline.DEBUG)

# save some test images

'''
heatmap = HeatmapContainer(shape=(10, 20))
heatmap.add_heatmap([((0,0),(10,20))])
heatmap.add_heatmap([((0,0),(5,5))])
heatmap.render_heatmap(clip_threshold=False)
'''

'''
trn = trainer.Trainer()
trn.look_for_classifier_type()
'''


'''
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

'''

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

scales = [
    # (scale, ystart, ystop, xstart, xstop)
    # (0.5, 388, 450, 564, 1076),
    (1, 400, 496, 504, 1272),
    (1.25, 368, 528, 512, 1280),
    (1.5, 368, 560, 512, 1280),
    (2, 368, 624, 512, 1280),
]

def simple_process_image(img):
    t1 = time.time()

    windows = image_features.slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop,
                                          xy_window=(64, 64), xy_overlap=(overlap, overlap))
    hot_windows = classifier.search_windows(img, windows, svc, X_scaler, color_space=color_space,
                                            spatial_size=spatial_size, hist_bins=hist_bins,
                                            orient=orient, pix_per_cell=pix_per_cell,
                                            cell_per_block=cell_per_block,
                                            hog_channel=hog_channel, spatial_feat=spatial_f,
                                            hist_feat=hist_f, hog_feat=hog_f)
    # print(time.time() - t1, 'Seconds to process image searching', len(windows), 'windows')

    return visualization.draw_boxes(img, hot_windows, color=(0, 0, 255), thick=3)

def process_heat_image(img, heatmap_history, render_heatmap=True, save_matches=False):
    windows = []
    confidence = []

    for s in scales:
        scale, ystart, ystop, xstart, xstop = s
        windows_s, confidence_s = classifier.find_cars(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, min_confidence=0.1)
        windows.extend(windows_s)
        confidence.extend(confidence_s)

    if save_matches:
        for idx, bbox in enumerate(windows):
            matched_subimage = img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
            visualization.save_image(matched_subimage, config.Pipeline.IMG_OUTPUT_DIR + 'matches_debug/' + str(heatmap_history.frame_counter).zfill(5)  + '_' + str(idx) + '.jpg')


    heatmap_history.add_to_heatmap(windows, confidence)
    heatmap_history.draw_labeled_bboxes(img)

    if render_heatmap:
        hm = heatmap_history.render_heatmap(thresholded=True)
        hm = cv2.resize(hm, (0, 0), fx=0.25, fy=0.25)
        img[0:hm.shape[0], img.shape[1]-hm.shape[1]:img.shape[1]] = hm

    return img

'''
for img_src in glob.glob(config.Pipeline.IMG_INPUT):
    img = visualization.load_image(img_src)
    window_img = process_image(img)
    visualization.save_image(window_img, config.Pipeline.IMG_OUTPUT_DIR + img_src)
'''


from moviepy.editor import VideoFileClip

input_video_file = "project_video.mp4"
output_video = "output_video/" + input_video_file

heatmap_obj = HeatmapContainer(over_frames=1, threshold=0)

clip = VideoFileClip(input_video_file).subclip('00:00:24.50', '00:00:24.75')# .subclip('00:00:24.50', '00:00:26.00')#.subclip('00:00:18.00', '00:00:20.00')
out_clip = clip.fl_image(lambda img: process_heat_image(img, heatmap_obj))
out_clip.write_videofile(output_video, audio=False)



'''
for img_src in glob.glob(config.Pipeline.IMG_INPUT):
    ystart = 400
    ystop = 656

    xstart = 416
    xstop = 1280

    scale = 1

    img = visualization.load_image(img_src)
    t1 = time.time()
    windows = classifier.find_cars(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    heat_image = np.zeros_like(img[:, :, 0]).astype(np.float)

    # out_img = visualization.draw_boxes(img, windows_1, (255, 0, 0), 4)

    # Add heat to each box in box list
    heat = image_features.add_heat(heat_image, windows)

    # Apply threshold to help remove false positives
    heat = image_features.apply_heat_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    out_img = visualization.draw_labeled_bboxes(np.copy(img), labels)

    print(time.time() - t1, 'Seconds to process image')

    visualization.save_image(out_img, config.Pipeline.IMG_OUTPUT_DIR + img_src)
    visualization.save_image(heatmap, config.Pipeline.IMG_OUTPUT_DIR + img_src + '_heat.jpg')

'''


for img_src in glob.glob(config.Pipeline.IMG_INPUT):
    hog_vis = True
    img = visualization.load_image(img_src)
    img_filename = os.path.basename(img_src)

    windows = []
    print(img_filename)
    for s in scales:
        scale, ystart, ystop, xstart, xstop = s
        px_scale = int(scale * 64)

        all_boxes = image_features.slide_window(img, [xstart, xstop], [ystart, ystop], (px_scale, px_scale), (0.875, 0.875))

        windows_s, confidence_s, hog_img = classifier.find_cars(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient, pix_per_cell,
                                         cell_per_block, spatial_size, hist_bins, hog_vis)

        print(px_scale, confidence_s)
        hog_combined = np.copy(img)
        hog_img = cv2.resize(hog_img, (0,0), fx=scale, fy=scale)
        hog_combined[ystart:ystart+hog_img.shape[0], xstart:xstart + hog_img.shape[1]] = hog_img


        out_img_s = visualization.draw_boxes(img, all_boxes, (255, 0, 0), 1)
        out_img_s = visualization.draw_boxes(out_img_s, windows_s, (0, 255, 0), 1)
        hog_combined = visualization.draw_boxes(hog_combined, windows_s, (0, 255, 0), 1)

        visualization.save_image(hog_combined, config.Pipeline.IMG_OUTPUT_DIR + img_filename +'_' + str(px_scale) + '_hog.jpg')
        visualization.save_image(out_img_s, config.Pipeline.IMG_OUTPUT_DIR + img_filename + '_' + str(px_scale) + '_boxes.jpg')

        windows.extend(windows_s)


    out_img = visualization.draw_boxes(img, windows, (0, 255, 255), 1)
    visualization.save_image(out_img, config.Pipeline.IMG_OUTPUT_DIR + img_filename + '_all_detected_boxes.jpg')
