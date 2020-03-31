from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Input, Flatten
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

x_train = np.load(sys.argv[1]).astype(float) / 255.0
y_train = np.load(sys.argv[2]).astype(int)
y_train_cat = keras.utils.to_categorical(y_train, 2)

x_test = np.load(sys.argv[3]).astype(float) / 255.0
y_test = np.load(sys.argv[4]).astype(int)
y_test_cat = keras.utils.to_categorical(y_test, 2)

print(x_train.shape)
print(np.sum(y_train), y_train.shape[0])

print(x_test.shape)
print(np.sum(y_test), y_test.shape[0])

# for i in range(x_test.shape[0]):
#   if y_test[i] == 0:
#     continue
#   cv2.imshow("image", x_test[i])
#   print(y_test[i])
#   if cv2.waitKey(0) > 0:
#       continue

model = keras.models.load_model('lenet.h5')
predictions = model.predict(x_test)

fp = 0
tp = 0
fn = 0
tn = 0

for i in range(predictions.shape[0]):
  p = predictions[i]
  if y_test[i] == 0: # true nonbird
    if p[0] >= p[1]: # true negative
      tn += 1
    else:            # false positive
      fp += 1
  else:              # true bird
    if p[1] >= p[0]: # true positive
      tp += 1
    else:            # false negative
      fn += 1

print(fp, tp, fn, tn)