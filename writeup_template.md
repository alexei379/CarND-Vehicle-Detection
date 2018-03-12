**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

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


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in function `get_hog_features` of the file called `image_features.py`. This function is called from `single_image_features` function to build a HOG feature vector from all color channels and stack them together with color spatial binning and histogram. I call `single_image_features` from `extract_features` function, that iterates over the list of training images and returns collection of feature vectors.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

| vehicle | non-vehicle |
| - | - |
| ![vehicle] | ![nonvehicle] |


During lectures I had a chance to explore different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) parameters. `pixels_per_cell=8` and `cells_per_block=2` seemed to work well and I wanted to explore more color spaces and number of orientations (see next section).

For demo purposes in this section I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space channel `Y` and HOG parameters `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)` and range of  `orientations=[6, 8, 12]`

| Type | `orientations=6` | `orientations=8` | `orientations=12` |
| - | - | - | - |
| ![vehicle] | ![car_hog_c0_o6] | ![car_hog_c0_o8] | ![car_hog_c0_o12] |
| ![nonvehicle] | ![noncar_hog_c0_o6] | ![noncar_hog_c0_o8] | ![noncar_hog_c0_o12] |


#### 2. Explain how you settled on your final choice of HOG, binned color and histograms of color feature parameters

I tried varous combinations of the parameters with the help of function `look_for_feature_params` in `trainer.py`. All it does it iterates over specified disctionaries with parameter options, trains the model and prints accuracy. I would add values I wanted to compare to appropriate lists and would analize the results. Detailed output results can be foud in `params_analysis.xlsx`

I trained the model (see details on training in next section) using variations of parameters on a subset of random 1000 samples and then confirmed the accuracy on the full training set for top choises.

First I explored various color spaces for HOG. I fixed HOG `orientations=9`, `pixels_per_cell=8` and `cells_per_block=2`, and was trying to select best color space and chanel. I got the initial results based on random 1000 training samples and performed training on full set for top 3:

| Color Space| Channel | Accuracy |
| -  | - | - |
| YUV	| ALL	| 0.9673 |
| LUV	| ALL	| 0.9645 |
| YCrCb	| ALL	| 0.9611 |

Next, I performed the same procedure to select best number of HOG orientations from [6, 7, 8, 9, 10] for the color spaces above. I got best results for `LUV` and `YCrCb` colorspaces using `ALL` chanels and `orientations=8`.

Using the same method I selected number of `bins=64` for color histogram and size of `spatial=(16, 16)` for spatial binning of color.

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

I perform the classifier training in `train` function in `trainer.py`. I implemented `extract_training_data` to extract training data using HOG, color histogram and binning features. I scale data using `StandardScaler`. I perform training using `test_size=0.1` split.
Trained classifier is pickeled for future use in pipeline.

I was exploring LinearSVC vs. SVC tuned with GridSearchCV (`look_for_classifier_type` in `trainer.py`). Best options for SVC were `{'kernel': 'rbf', 'gamma': 0.0001, 'C': 10}`, but I ended up using LinearSVC as it performed better on project video.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided that searching different areas of image with different window size might give a good result a be more optimal then searching multiple scales over the whole image, as detections further away would be better matched by the small windows and closer to the camera - by larger windows. I decided to have a big overlap (87.5%) to get more detections.

I explored various options in `exploration.py` lines 94-126 and came up with the following grids that prooved to be working fine in the video pipeline.

| Window | Sample grid with detections |
| - | - |
| 32x32 | ![grid_32] |
| 64x64 | ![grid_64] |
| 80x80 | ![grid_80] |
| 96x96 | ![grid_96] |
| 128x128 | ![grid_128] |

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

