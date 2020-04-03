import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
from tensorflow import keras


# load bird data

x_bird_train = np.load(sys.argv[1]).astype(float) / 255.0
y_bird_train = np.load(sys.argv[2]).astype(int)
print(x_bird_train.shape, y_bird_train.shape)
x_bird_test = np.load(sys.argv[3]).astype(float) / 255.0
y_bird_test = np.load(sys.argv[4]).astype(int)

# load aircraft data

# x_plane_train_1 = np.load(sys.argv[5]).astype(float)
# y_plane_train_1 = np.load(sys.argv[6]).astype(int)
# x_plane_train_2 = np.load(sys.argv[7]).astype(float)
# y_plane_train_2 = np.load(sys.argv[8]).astype(int)
# x_plane_train_3 = np.load(sys.argv[9]).astype(float)
# y_plane_train_3 = np.load(sys.argv[10]).astype(int)

# x_plane_test_1 = np.load(sys.argv[11]).astype(float)
# y_plane_test_1 = np.load(sys.argv[12]).astype(int)
# x_plane_test_3 = np.load(sys.argv[13]).astype(float)
# y_plane_test_3 = np.load(sys.argv[14]).astype(int)
x_plane_train = np.load(sys.argv[5]).astype(float)
y_plane_train = np.load(sys.argv[6]).astype(int)
print(x_plane_train.shape, y_plane_train.shape)
# load negative examples
x_plane_train_neg_1 = np.load(sys.argv[7]).astype(float)
y_plane_train_neg_1 = np.load(sys.argv[8]).astype(int)
x_plane_train_neg_2 = np.load(sys.argv[9]).astype(float)
y_plane_train_neg_2 = np.load(sys.argv[10]).astype(int)
x_plane_train_neg_3 = np.load(sys.argv[11]).astype(float)
y_plane_train_neg_3 = np.load(sys.argv[12]).astype(int)
print(x_plane_train_neg_1.shape, y_plane_train_neg_1.shape)
print(x_plane_train_neg_2.shape, y_plane_train_neg_2.shape)
print(x_plane_train_neg_3.shape, y_plane_train_neg_3.shape)

# combine aircraft data only:

# x_train = np.append(x_plane_train_2, x_plane_train_3, axis=0)
x_train = np.append(x_plane_train, x_plane_train_neg_1, axis=0)
x_train = np.append(x_train, x_plane_train_neg_2, axis=0)
x_train = np.append(x_train, x_plane_train_neg_3, axis=0)

# y_train = np.append(y_plane_train_2, y_plane_train_3)
y_train = np.append(y_plane_train, y_plane_train_neg_1)
y_train = np.append(y_train, y_plane_train_neg_2)
y_train = np.append(y_train, y_plane_train_neg_3)

# x_test = np.append(x_plane_test_1, x_plane_test_3, axis=0)
# y_test = np.append(y_plane_test_1, y_plane_test_3)

# combine all training data

x_train = np.append(x_bird_train, x_plane_train, axis=0)
# x_train = np.append(x_train, x_plane_train_neg_1, axis=0)
# x_train = np.append(x_train, x_plane_train_neg_2, axis=0)
# x_train = np.append(x_train, x_plane_train_neg_3, axis=0)

y_train = np.append(y_bird_train, y_plane_train)
# y_train = np.append(y_train, y_plane_train_neg_1)
# y_train = np.append(y_train, y_plane_train_neg_2)
# y_train = np.append(y_train, y_plane_train_neg_3)

# print(x_bird_train.shape, y_bird_train.shape)
# print(x_plane_train_1.shape, y_plane_train_1.shape)
# print(x_plane_train_2.shape, y_plane_train_2.shape)
# print(x_plane_train_3.shape, y_plane_train_3.shape)
print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

# verify combine
# for i in range(x_train.shape[0]):
#     if y_train[i] == 2:
#         cv2.imshow("image", x_train[i])
#         print(y_train[i])
#         cvk = cv2.waitKey(0)
#         if cvk & 0xFF == ord('q'):
#             break
#         elif cvk & 0xFF == ord('v'):
#             print(x_train[i])
#         else:
#             continue

# save data
print(np.count_nonzero(y_train))
np.save("x_train_combined.npy", x_train)
np.save("y_train_combined.npy", y_train)
# np.save("x_train_aircraft.npy", x_train)
# np.save("y_train_aircraft.npy", y_train)