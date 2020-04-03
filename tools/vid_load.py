import numpy as np
import cv2
import matplotlib.pyplot as plt
import os.path
from os import path
import signal
import keyboard

a = 0
b = 0

def keyboardInterruptHandler(signal, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(signal))
    exit(0)



# read video
cap = cv2.VideoCapture('/home/subt/Downloads/yolo_output_1_3.avi')


c = 0
count = 0
bird = 0
plane = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    c = c+1
    if(path.exists("/home/subt/workspaces/ML_Project/cvpr15/videos/planes/Video_1_3.avi-labels/"+str(c)+".label")):
        f = open("/home/subt/workspaces/ML_Project/cvpr15/videos/planes/Video_1_3.avi-labels/"+str(c)+".label", "r")
        l1 = f.readline()
        l2 = f.readline()
        if(len(l2) != 0):
            count = count+1
        count = count+1
    print("Total groundtruth: " + str(count))


    # Our operations on the frame come here
    
    if keyboard.is_pressed('p'):
        plane = plane + 1
        print("plane: ", plane)
    elif keyboard.is_pressed('b'):
        bird = bird + 1
        print("bird: ", bird)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(0) & 0xFF == ord('s'):
        continue

cv2.waitKey(0)
if cv2.waitKey(0) > 0:
    pass