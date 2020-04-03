import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2

train_test_split = 0.9
x_data = np.load(sys.argv[1]).astype(float)
y_data = np.load(sys.argv[2]).astype(int)

print("Input data shape", x_data.shape, y_data.shape)

idx = np.arange(x_data.shape[0])
np.random.shuffle(idx)
train_samples = int(train_test_split * x_data.shape[0])
train_idx = idx[:train_samples]
test_idx = idx[train_samples:]

x_train, x_test = x_data[train_idx, :, :, :], x_data[test_idx, :, :, :]
y_train, y_test = y_data[train_idx], y_data[test_idx]

print("Train data shape", x_train.shape, y_train.shape)
print("Test data shape", x_test.shape, y_test.shape)

# for i in range(x_train.shape[0]):
#     if y_train[i] == 0:
#         cv2.imshow("image", x_train[i])
#         print(y_train[i])
#         if cv2.waitKey(0) > 0:
#             continue

np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
np.save("x_test.npy", x_test)
np.save("y_test.npy", y_test)