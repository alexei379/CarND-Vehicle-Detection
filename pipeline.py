import config
import utils
import classifier
import cv2
from heatmap_container import HeatmapContainer
from moviepy.editor import VideoFileClip

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

X_scaler = utils.load_classifier(config.Classifier.SCALER_FILE)
svc = utils.load_classifier(config.Classifier.CLS_FILE)


def process_heat_image(img, heatmap_history, render_heatmap=True, save_matches=False):
    windows = []

    for s in scales:
        scale, ystart, ystop, xstart, xstop = s
        windows_s = classifier.find_cars(img,
                                         ystart, ystop, xstart, xstop, scale,
                                         svc, X_scaler,
                                         orient, pix_per_cell, cell_per_block,
                                         spatial_size, hist_bins,
                                         min_confidence=config.Pipeline.MIN_CONFIDENCE)
        windows.extend(windows_s)

    if save_matches:
        for idx, bbox in enumerate(windows):
            matched_subimage = img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
            utils.save_image(matched_subimage,
                             config.Pipeline.IMG_OUTPUT_DIR + 'matches_debug/'
                             + str(heatmap_history.frame_counter).zfill(5) + '_' + str(idx) + '.jpg')

    heatmap_history.add_to_heatmap(windows)
    heatmap_history.draw_labeled_bboxes(img)

    # overlay heatmap preview
    if render_heatmap:
        hm = heatmap_history.render_heatmap(thresholded=True)
        hm = cv2.resize(hm, (0, 0), fx=0.25, fy=0.25)
        img[0:hm.shape[0], img.shape[1] - hm.shape[1]:img.shape[1]] = hm

    return img


input_video_file = "project_video.mp4"
output_video = "output_video/" + input_video_file

heatmap_obj = HeatmapContainer()

clip = VideoFileClip(input_video_file)
# .subclip('00:00:05.00', '00:00:07.00')
# .subclip('00:00:18.00', '00:00:20.00')
# .subclip('00:00:23.50', '00:00:26.00')
# .subclip('00:00:37.50', '00:00:39.00')
# .subclip('00:00:44.00', '00:00:47.00')
# .subclip('00:00:47.50')

out_clip = clip.fl_image(lambda img: process_heat_image(img, heatmap_obj))
out_clip.write_videofile(output_video, audio=False)
