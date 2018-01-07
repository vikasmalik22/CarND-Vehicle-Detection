# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project
---
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it! The code for the project is contained in python notebook Vehicle_Detection_Tracking.ipynb

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images. The code for this is contained in **code cell 2**. Here is an example of randomly selected `vehicle` and `non-vehicle****` classes:

![cars](https://github.com/vikasmalik22/CarND-Vehicle-Detection/blob/master/output_images/cars.png)

![not-cars](https://github.com/vikasmalik22/CarND-Vehicle-Detection/blob/master/output_images/not-cars.png)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9,10,11,12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![Hog](https://github.com/vikasmalik22/CarND-Vehicle-Detection/blob/master/output_images/Hog.png)

The code for extracting HOG features from an image is defined by the method *get_hog_features* in code cell 5.

I also used **Spatial Binning** to retrieve some relevant features of the training data. While it could be cumbersome to include three color channels of a full resolution image, we can perform spatial binning on an image and still retain enough information to help in finding vehicles. The coce for extracting spatial binning features is contained in method ***bin_spatial*** in code cell 7.

Here is an example of using spatial binning by using size of (32,32) on an original image of size (64,64). This results in a feature vector of length 3072.

![spatial_binning](https://github.com/vikasmalik22/CarND-Vehicle-Detection/blob/master/output_images/spatial_binning.png)

I also explored and used **Histograms of Color** method to extract features. It helps in differentiating the images by the intensity and range of color distribution. This makes it good comparator in differentiating between car and non-car images. 

The method **extract_features** in the code cell 10 accepts a list of image paths,  HOG parameters (as well as one of a variety of destination color spaces, to which the input image is converted), spatial dimensions, number of histogram bins and produces a flattened array of combined (Hog, Spatial and histogram) features for each image in the list.

I defined all the parameters to do this extraction in code cell 11 under section titled "Feature Extraction Parameters". And then I extract all the features under section titled "Extract Features for Input Datasets" in code cell 12.
I combine the features and use StandardScaler function to normalize the feature vector under section titled "Combine the Features" in code cell 13.
Save/dump the scaler in pickle file to load and use later so that I don't have to run the Scale and extract the features again. This is done in code cell 14 under section titled "Dump/Save the Scaler".
The features and labels are then shuffled and split into training and test sets in preparation to be fed to a linear support vector machine (SVM) classifier. This is done under section titled "Shuffle and Split the data in training and test & Train the Classifier" in code cell 15.
I then save the Classifier in the pickle file for later use. This is done in code cell 16.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters of Hog, Spatial Bining and Histogram of Bins. The table below documents below the 25 different parameter combinations I tested and explored.

![Result](https://github.com/vikasmalik22/CarND-Vehicle-Detection/blob/master/output_images/Parameter_Results.PNG)

First, I explored the colorspace **YCrCb** since this is what we used in the lesson and was giving good results. The first 12 combinations are in green color because here I only used every other second car image as many of the car images were repeating in the dataset given. I kept all the HOG channels and tried different combinations of orientations, pixels per cell, cells per block and spatial size and histogram bins. Even though many of these combinations give high accuracy but when I tried on running video there were many false positives and results were not upto the mark. The best results were given for combination number 9 and 10 with orientation 9 and 12. 

Then, I explored the results with other color spaces which are shown in blue color from cobination number 13 to 25 but this time using all the car images. That's why the car feature extraction is double than for those in green color. When not using the spatial binning and histogram bins (combination number 20-25) the highest accuracy came for colorspaces YCrCb and YUV. The accuracy for both the colorspaces were almost the same but training time for YCrCb is lesser. 

I then further wanted to check and see if the accuracy can be improved further by using spatial binnning and histogram bins (combination number 13-19) the highest accuracy and least amount of training time came for colorspace YCrCb as compared to the other colorspaces. 

And this way, I finally narrowed down to using the combination number 13. I didn't use the combination number 14 even it has better accuracy because it gave me almost the same accuracy as 13 but has higher training time, prediction time and larger feature vector.

These parameters are present in code cell 11 under section titled "Feature Extraction Parameters".

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the section titled "Shuffle and Split the data in training and test & Train the Classifier" I trained a linear SVM with the default classifier parameters and used HOG features, spatial binning and histogram of colors feature. This was I was able to achieve a test accuracy of 99.10 %.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Until now, we feed a classifier with an 64 x 64 pixels image and get a result from it: car or non-car. In order to do this for an entire image (720 x 1280), we use a sliding window. I cropped the images with the area of interest in which I am hopefull of finding the car images. 

The Image below shows the cropped Region of Interest (ROI) in YCrCb colorspace.

![cropped](https://github.com/vikasmalik22/CarND-Vehicle-Detection/blob/master/output_images/cropped.PNG)

Then sliced the image in small frames, resized it to the right size (64x64), and applied the classification algorithm.

In the section titled "To find cars in an image" the method **find_cars** is present in the code cell 18, which combines feature extraction with a sliding window search, but rather than performing feature extraction on each window individually which can be time consuming, the features are extracted for the ROI.

And then these full-image features are subsampled according to the size of the window and then fed to the classifier. The method performs the classifier prediction on the combined features for each window region and returns a list of rectangle objects corresponding to the windows that generated a positive ("car") prediction.

The below images shows the attempts at using find_cars on one of the test images, using a different window sizes.Since, the car can appear in different sizes. I applied different windows sizes over the image i.e. scale value of 1.1, 1.4 and 1.6. This is done in code cell 20, 21, 22.

![find_cars1](https://github.com/vikasmalik22/CarND-Vehicle-Detection/blob/master/output_images/find_cars1.png)

![find_cars2](https://github.com/vikasmalik22/CarND-Vehicle-Detection/blob/master/output_images/find_cars14.png)

![find_cars3](https://github.com/vikasmalik22/CarND-Vehicle-Detection/blob/master/output_images/find_cars18.png)


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In the end, I finalised and used the following scale values.
scales = [1.1, 1.4, 1.6, 2.0, 2.2, 2.4, 3.0]

Following is a result of combining all the different window sizes using the above scale values.

![find_carsfinal](https://github.com/vikasmalik22/CarND-Vehicle-Detection/blob/master/output_images/find_cars_final.png)

Next, I used the technique mentioned in chapter 37 on how to remove false positives from multiple detections. A true positive is usually consist of many positive detections, whereas false positives are typically accompanied by only one or two detections. a combined heatmap and threshold is used to differentiate between the two. The add_heat function simply adds +1 for all pixels within windows where a positive detection is reported by the classifier. Areas enclosed by more overlapping rectangles are assigned higher levels of heat. The following image is the resulting heatmap from the detections in the image above:

![apply_heat](https://github.com/vikasmalik22/CarND-Vehicle-Detection/blob/master/output_images/apply_heat.PNG)

A threshold is applied to the heatmap using value of 1, setting all pixels that don't exceed the threshold to zero. The result is shown below in the image.

![heat_threshold](https://github.com/vikasmalik22/CarND-Vehicle-Detection/blob/master/output_images/heat_threshold.PNG)

The scipy.ndimage.measurements.label() function collects spatially contiguous areas of the heatmap and assigns each a label:

![heat_threshold](https://github.com/vikasmalik22/CarND-Vehicle-Detection/blob/master/output_images/labels.PNG)

And the final detection area we take labels image and put bounding boxes around the labeled regions.:

![heat_threshold](https://github.com/vikasmalik22/CarND-Vehicle-Detection/blob/master/output_images/final_box.PNG)

The final implementation performs very well, identifying the vehicles in each of the images with no false positives.

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used a class Windows to store the rectangles where the cars were detected during each frame of the video using find_cars(). The code for this is contained under the section titled "Windows Class to store the rectangles where the cars were detected during each frame". This class stores all the detections for each frame and keep only the last 10 frames of the video. Detections are stored in the class using add_win() and keeps only last 10 frames. Then we use these last 10 frames and perform heatmap, threshold and labels on them combined instead of doing on each frame individually. I used the threshold value for the heatmap is set to 1 + len(win_pos)//2. I found this value through experiments running video for few frames and checking the result.

The code for the pipeline processing of the image processing is contained in section titled "Combining all together in Pipeline function to process Image". 

Additionally, I also created a heatmap of the frames which is combined together with the final output and the code for the same is contained in the same section.

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

The main problem was to find the classifier with apporpriate parameters which have good detection accuracy. It took some time to find the right parameters for the classifier which have high accuracy and also takes least execution time. I found the combination of HOG, spatial binning and Histogram of colors give good and accurate results.

The sliding window method is expensive, it took 13-14 mins to process video of 50 seconds. For a real-time application, we need an optimized solution like parallel processing.

The ideal solution would be to have a very high accuracy classifier which makes prediction in no time and without the need to use previous detections. This kind of real-time solution would require large computing power and parallel processing.

The pipeline is trained only to detect car and non-car images on the dataset provided which would not be sufficient to detect vehicles not present in that dataset.

The pipeline can most likely fail where there are small window scales because they produce more false positives. The oncoming cars are the issue in this case and also distant cars.

I think given the time, I would want to use the neural network approach to detect the cars. 






