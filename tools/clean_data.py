import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2


x_data = np.load(sys.argv[1]).astype(float)
y_data = np.load(sys.argv[2]).astype(int)
print(x_data.shape, y_data.shape)

idx = []
for i in range(x_data.shape[0]):
    if np.isnan(np.sum(x_data[i])):
        continue
    else:
        idx.append(i)

x_data_clean = x_data[idx, :, :, :]
y_data_clean = y_data[idx]
print(x_data_clean.shape, y_data_clean.shape)

np.save("x_data_clean.npy", x_data_clean)
np.save("y_data_clean.npy", y_data_clean)