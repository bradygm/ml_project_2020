# Flying Object Detection and Classification
 10701 Introduction to Machine Learning. Project members Sourish Ghosh, Mohammadreza Mousaei, and Brady Moon.

## Dependencies
- Tensorflow v2.1.0
- OpenCV

## Running our code
The full pipeline can be executed by running:
```bash
$ python detector/main.py [path/to/video/file.mp4] [path/to/saved_model.h5]
```
Please change `image_size` variable in the `main.py` file to match the video resolution. 

# Scripts breakdown
Our code is separated into three folders: tools, detector, and classifier. 

## Tools
The tools folder contains helper scripts for processing our data. This includes scripts for cleaning, combining, augmenting, splitting, and loading data. 

## Detector
Two adjacent frames and passes to the `find_moving_objects` function found within the `detect_objects.py`file. This script finds BRIEF descriptors in each image, matches them, and then passes them to the `cv2.findHomography` function. We previously used our own RANSAC and homography code, but ran into odd edge cases. It then warps one of the images using the homography, takes the absolute difference between them, and checks them against a threshold. Our default value for this threshold is 20. We then perform a series of binary erosions and dilations. We then find the bounding boxes for the blobs and return the warped image and bounding boxes. 

## Classifier
The combined dataset used to train our classifier can be downloaded from here: [https://drive.google.com/open?id=1Zb2C_KU4fudB8X3c8SvYfzDX1350TKK3](https://drive.google.com/open?id=1Zb2C_KU4fudB8X3c8SvYfzDX1350TKK3)
Once the data is downloaded, you can train the classifier by executing:
```bash
$ python classifier/train_classifier.py [path/to/x_train.npy] [path/to/y_train.npy] [path/to/x_test.npy] [path/to/y_test.npy]
```
After the classifier is train, the model will be saved in the `models` folder. This saved model will be used during the execution of the overall pipeline.
