# Flying Object Detection and Classification
 10701 Introduction to Machine Learning. Project members Sourish Ghosh, Mohammadreza Mousaei, and Brady Moon.
## Running our code
Sourish add details here for running the full pipeline. 

# Scripts breakdown
Our code is separated into three folders: tools, detector, and classifier. 

## Tools
The tools folder contains helper scripts for processing our data. This includes scripts for cleaning, combining, augmenting, splitting, and loading data. 

## Detector
You can run the detector code on its own and see the output by running the `main.py` file with path to your video file as the input argument. It initializes a `pipeline` variable, loads the video with the `load_video` script, and then finds bounding boxes with the `detect_bbox` script. Within the `detect_bbox` script, it loads two adjacent frames and passes them to the `find_moving_objects` function found within the `detect_objects.py`file. This script finds BRIEF descriptors in each image, matches them, and then passes them to the `cv2.findHomography` function. We previously used our own RANSAC and homography code, but ran into odd edge cases. It then warps one of the images using the homography, takes the absolute difference between them, and checks them against a threshold. Our default value for this threshold is 7. We then perform a series of binary erosions and dilations. We then find the bounding boxes for the blobs and return the warped image and bounding boxes. 

## Classifier
Sourish
