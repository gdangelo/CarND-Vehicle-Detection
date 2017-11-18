## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./figures/car_not_car.png
[image2]: ./figures/HOG_car_example.png
[image3]: ./figures/HOG_non_car_example.png
[image4]: ./figures/sliding_windows_0_7.jpg
[image5]: ./figures/sliding_windows_1_0.jpg
[image6]: ./figures/sliding_windows_1_5.jpg
[image7]: ./figures/sliding_windows_2_0.jpg
[image8]: ./output_images/test1_detected_windows.jpg
[image9]: ./output_images/test1_heatmap.jpg
[image10]: ./output_images/test1_car_position.jpg
[image11]: ./output_images/test2_car_position.jpg
[image12]: ./output_images/test3_car_position.jpg
[image13]: ./output_images/test4_car_position.jpg
[image14]: ./output_images/test5_car_position.jpg
[image15]: ./output_images/test6_car_position.jpg
[video1]: ./output_videos/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! Please find the code for the project in the `vehicle_detection.py` file.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images from the two provided datasets ([vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Car & Non Car][image1]

The code for extracting features (binary spatial, color histograms, and HOG) from an image is defined by the method `extract_features` in lines 49 through 87. And each features set is extracted in the corresponding methods called `bin_spatial` (in lines 21 through 26), `color_hist` (in lines 29 through 36), and `get_hog_features` (in lines 39 through 45).

Before extracting features, the input images have been converted from `RGB` to `YCrCb`. The choice of the color space has been made after a number of trials between `RGB`, `HSV`, `YUV`, `LUV`, and `YCrCb` color spaces.

The HOG features have been extracted using the `hog` method from `skimage.feature` for the 3 color channels with the following parameters: `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.

The figure below shows the associated HOG features for car and non-car images on the 3 color channels.

![HOG features for car][image2]

![HOG features for non car][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and I settled on my final choice of HOG parameters based upon the performance of the SVM classifier. The accuracy was the main driver. I've tried to reduce as possible the complexity to keep a good accuracy (around 98-99%) and train the model in a reasonable amount of time.

The final HOG parameters chosen were the following: YUV colorspace, 9 orientations, 8 pixels per cell, 2 cells per block, and ALL channels of the colorspace.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a SVM classifier using with the default settings and using the binary spatial, color histograms, and HOG features. I was able to achieve a test accuracy of 98.8%. The code for the training is located in lines 309 through 353 in the `train_svc` method.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this section can be found in lines 174 through 261 in the `find_cars` method.

The method first extract the same features as the ones that has been extracted for training the classifier. Expect that the HOG features are extracted for the entire region of interest (above 400 pixels in y coordinates). It is less time consuming than performing the HOG extraction on each window individually.

After the features extraction, the image is subsampled in 64x64 windows and fed to the classifier. Hence, the method returns the bounding boxes for the windows that have generated a positive prediction. The `decision_function` from the SVC classifier is also used to only retain predictions with a certain confidence (measured as the distance of the samples to the separating hyperplane):

```python
test_prediction = clf.predict(test_features)
test_confidence = clf.decision_function(test_features)

if test_prediction == 1 and test_confidence > 0.4:
          xbox_left = np.int(xleft*scale)
          ytop_draw = np.int(ytop*scale)
          win_draw = np.int(window*scale)
          bboxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
```

I explored several configurations of windows scale and different region of interest for each scale. The following four images show the configurations of all windows for small (0.7x), medium (1.0x, 1.5x), and large (2.0x) scales:

![Sliding windows][image4]

![Sliding windows][image5]

![Sliding windows][image6]

![Sliding windows][image7]

Hence, the pipeline calls the `find_cars` method for each window scale and the bounding boxes returned are combined. The image below show the bounding boxes obtained with this four scale configurations.

We can notice that there are several positive predictions on each car, and a positive prediction on the car in the oncoming lane (which is great!).

![Bounding boxes][image8]

Based on the resulting bounding boxes a heat map is built in the `add_heat` method. On a black image, each pixel is incremented by 1 every time it is contained in a bounding boxe. Hence, areas with more overlapping bounding boxes have higher values than the ones with one or few bounding boxes.

The following image is the resulting heatmap from the detections in the image above:

![Heat map][image9]

Then a threshold is applied to the heatmap in order to remove (setting to 0) the pixels that don't exceed the threshold. I choose to set that threshold to 1 in order to keep the detected vehicles in the oncoming lane.

Finally I used the `scipy.ndimage.measurements.label()` method to find out the extremities of each car based on the heatmap above. Here's the result:

![Car position][image10]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Below you could find the result of applying my pipeline to the test images provided for this project:

![Test image][image10]

![Test image][image11]

![Test image][image12]

![Test image][image13]

![Test image][image14]

![Test image][image15]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](https://github.com/gdangelo/CarND-Vehicle-Detection/blob/master/output_videos/project_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

See section "Sliding Windows Search 1." above for all the details.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problems that I faced while implementing this project were mainly concerned with balancing detection accuracy and windows scale to avoid as much as possible false positive and keep the tracking vehicles along the video.

While the accuracy of the classifier is pretty good on the test dataset (98.8%) it could be limited in different weather conditions. Also the processing speed is very slow and could be improved a lot using another detection techniques.

Hence, the following improvements could be made to improve my pipeline:
- Train the SVC with more data (use the [driving Udacity dataset](https://github.com/udacity/self-driving-car/tree/master/annotations))
- Adjust the hyperparameters of the pipeline to reduce as much as possible false positives
- Use a CNN classifier such as R-CNN, Fast R-CNN, Faster R-CNN, or YOLO
- Combine lane finding and vehicle detection pipeline
