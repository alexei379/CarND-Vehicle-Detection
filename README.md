**Vehicle Detection Project**

The goals/steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

While implementing this project I was using course materials, Q&A session video and discussion forums provided by Udacity.

[//]: # (Image References)
[vehicle]: ./output_images/hod_demo/7_1_car.png
 [nonvehicle]: ./output_images/hod_demo/8_3_noncar.png

[car_hog_c0_o6]: ./output_images/hod_demo/7_2_car_hog_c0_o6.png
[car_hog_c0_o8]: ./output_images/hod_demo/7_2_car_hog_c0_o8.png
[car_hog_c0_o12]: ./output_images/hod_demo/7_2_car_hog_c0_o12.png
[noncar_hog_c0_o6]: ./output_images/hod_demo/8_3_noncar_hog_c0_o6.png
[noncar_hog_c0_o8]: ./output_images/hod_demo/8_3_noncar_hog_c0_o8.png
[noncar_hog_c0_o12]: ./output_images/hod_demo/8_3_noncar_hog_c0_o12.png

[grid_32]: ./output_images/test6.jpg_32_boxes.jpg
[grid_64]: ./output_images/test6.jpg_64_boxes.jpg
[grid_80]: ./output_images/test6.jpg_80_boxes.jpg
[grid_96]: ./output_images/test6.jpg_96_boxes.jpg
[grid_128]: ./output_images/test6.jpg_128_boxes.jpg

[hog_32]: ./output_images/test6.jpg_32_hog.jpg
[hog_64]: ./output_images/test6.jpg_64_hog.jpg
[hog_80]: ./output_images/test6.jpg_80_hog.jpg
[hog_96]: ./output_images/test6.jpg_96_hog.jpg
[hog_128]: ./output_images/test6.jpg_128_hog.jpg

[all_detected_boxes4]: ./output_images/test4.jpg_all_detected_boxes.jpg
[all_detected_boxes3]: ./output_images/test3.jpg_all_detected_boxes.jpg
[all_detected_boxes6]: ./output_images/test6.jpg_all_detected_boxes.jpg

[heatmap_demo]: ./output_images/heatmap_demo.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in function `get_hog_features` of the file called `image_features.py`. This function is called from `single_image_features` function to build a HOG feature vector from all color channels and stack them together with color spatial binning and histogram. I call `single_image_features` from `extract_features` function, that iterates over the list of training images and returns collection of feature vectors. I used labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) to train my classifier.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

| vehicle | non-vehicle |
| - | - |
| ![vehicle] | ![nonvehicle] |


During lectures I had a chance to explore different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) parameters. `pixels_per_cell=8` and `cells_per_block=2` seemed to work well and I wanted to explore more color spaces and number of orientations (see next section).

For demo purposes in this section, I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` the output looks like.

Here is an example using the `YCrCb` color space channel `Y` and HOG parameters `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)` and range of  `orientations=[6, 8, 12]`

| Type | `orientations=6` | `orientations=8` | `orientations=12` |
| - | - | - | - |
| ![vehicle] | ![car_hog_c0_o6] | ![car_hog_c0_o8] | ![car_hog_c0_o12] |
| ![nonvehicle] | ![noncar_hog_c0_o6] | ![noncar_hog_c0_o8] | ![noncar_hog_c0_o12] |


#### 2. Explain how you settled on your final choice of HOG, binned color and histograms of color feature parameters

I tried various combinations of the parameters with the help of function `look_for_feature_params` in `trainer.py`. All it does it iterates over specified dictionaries with parameter options, trains the model and prints accuracy. I would add values I wanted to compare to appropriate lists and would analyze the results. Detailed output results can be found in `params_analysis.xlsx`

I trained the model (see details on training in next section) using variations of parameters on a subset of random 1000 samples and then confirmed the accuracy of the full training set for top choices.

First I explored various color spaces for HOG. I fixed HOG `orientations=9`, `pixels_per_cell=8` and `cells_per_block=2`, and was trying to select best color space and channel. I got the initial results based on random 1000 training samples and performed training on the full set for top 3:

| Color Space| Channel | Accuracy |
| -  | - | - |
| YUV    | ALL    | 0.9673 |
| LUV    | ALL    | 0.9645 |
| YCrCb    | ALL    | 0.9611 |

Next, I performed the same procedure to select the best number of HOG orientations from [6, 7, 8, 9, 10] for the color spaces above. I got best results for `LUV` and `YCrCb` colorspaces using `ALL` channels and `orientations=8`.

Using the same method I selected the number of `bins=64` for color histogram and size of `spatial=(16, 16)` for spatial binning of color.

I tried various combinations of `[HOG, histogram, spatial]` features and classifier worked best with all 3 present.

Although during training `LUV` and  `YCrCb` colorspaces both performed well, I got better results in project video using `YCrCb` 

I stored the resulting parameters in `config.py` under `Classifier` section:
```python
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
```


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I perform the classifier training in `train` function in `trainer.py`. I implemented `extract_training_data` to extract training data using HOG, color histogram, and binning features. I scale data using `StandardScaler`. I perform training using `test_size=0.1` split.
Scaler and trained classifier are pickled for future use in the pipeline.

I was exploring LinearSVC vs. SVC tuned with GridSearchCV (`look_for_classifier_type` in `trainer.py`). Best options for SVC were `{'kernel': 'rbf', 'gamma': 0.0001, 'C': 10}`, but I ended up using LinearSVC as it performed better on project video.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided that searching different areas of an image with different window size might give a good result a be more optimal then searching multiple scales over the whole image, as detections further away would be better matched by the small windows and closer to the camera - by larger windows. I decided to have a big overlap (87.5%) to get more detections.

I explored various options in `exploration.py` lines 94-126 and came up with the following grids that proved to be working fine in the video pipeline. `slide_window` in `image_features.py` generates the grid with all possible windows, but I use it just for testing purpose.

Actual classifier implemented in `find_cars` in `classifier.py` (see details below).

| Size (Scale) | Sample grid with detections |
| - | - |
| 32x32 (0.5) | ![grid_32] |
| 64x64 (1)  | ![grid_64] |
| 80x80 (1.25)  | ![grid_80] |
| 96x96 (1.5)  | ![grid_96] |
| 128x128 (2)  | ![grid_128] |

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

`find_cars` in `classifier.py` scales the whole image, gets HOG features once per frame and then slides the scaled window to get build feature vector for the classifier. It allows performing HOG feature extraction just once for the whole image at each scale instead of doing it for each window, which results in a performance increase. It also adds spatially binned color (`bin_spatial` in `image_features.py`) and histograms of color (`color_hist` in `image_features.py`) to the feature vector. It calls trained LinearSVC and if the confidence is above `min_confidence=0.5`, the bounding boxes are added to the results. I use `draw_boxes` in `utils.py` to visualize various boxes on images.

Here are the samples of scaled HOG detection overlayde over the test images:

| Size (Scale) | Sample grid HOG overlay |
| - | - |
| 32x32 (0.5) | ![hog_32] |
| 64x64 (1)  | ![hog_64] |
| 80x80 (1.25)  | ![hog_80] |
| 96x96 (1.5)  | ![hog_96] |
| 128x128 (2)  | ![hog_128] |


Ultimately I searched on 5 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![all_detected_boxes4]
![all_detected_boxes3]
![all_detected_boxes6]


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
https://www.youtube.com/watch?v=cHjV4XDjD1A

[![](https://img.youtube.com/vi/cHjV4XDjD1A/0.jpg)](https://www.youtube.com/watch?v=cHjV4XDjD1A)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Video pipeline is implemented in `pipeline.py`. It gets positive detections using `find_cars` in `classifier.py` at various window scales, adds the windows to heatmap, get identified windows back from heatmap and renders result over the frame. I also added heatmap visualization in the top right corner of the video.

The heatmap is implemented in `heatmap_container.py` file and is used to combine multiple detections across several frames and eliminate false positives using threshold calculated over several frames.

I recorded the positions of positive detections in each frame of the video in `add_to_heatmap` function. I applied "boost" to the detected boxes if they are within proximity of previously detected blobs or are appearing from the side (not the middle) if detection region. I keep track if last 12 frame's heatmaps using `dequeue` structure. I summed up all heatmaps and then thresholded that map to identify vehicle positions. "Boosting" and thresholding the result over several frames to eliminate false positives worked quite well for the project video.

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected and drawn them in `draw_labeled_bboxes`.

I rendered the resulting labels and heatmap to the output video and they demonstrate both steps well. Example video frame:
![heatmap_demo]

---

### Discussion

#### 1. Briefly discuss any problems/issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline runs pretty slow on my laptop (0.5 FPS). Main areas of improvement I see are:
* try computing HOG for channels in parallel.
* experiment with windows that overlap less and with fewer scales, as combined with the heatmap the result might be comparable.
* I had very few false positives (which is good), but I took me a while to figure out a way how to get rid of them. I came up with "boosting" solution that allowed to have a higher threshold, but there might be more options to explore.
* Deep neural networks might also work nicely for this project.

