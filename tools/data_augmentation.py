import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import struct
import sys
import skimage.transform as tr
import skimage.util as ut


class Classify():
    def __init__(self, ):
        self.npy_data_X = []
        self.npy_data_Y = []
        self.filenames = []
        self.file_annotations = []
        self.X_data = []
        self.y_data = []
        self.train_cropped_images_filenames =[]
        self.train_cropped_images =[]
        self.train_cropped_labels =[]
        self.test_cropped_images_filenames =[]
        self.test_cropped_images =[]
        self.test_cropped_labels =[]
        self.raw_images = []


    # def read_video(self, addr):
    #     cap = cv2.VideoCapture(addr)
    #     ret, frame = cap.read()    
    #     #plt.figure()
    #     #plt.imshow(frame)

    def read_npy(self, addr):
        return np.load(addr)

    def read_annotations(self, addr):
        with open(addr) as openfileobject:
            tmp = []
            for line in openfileobject:
                if( line[0] == 'I'):
                    self.filenames.append(line)
                    if(len(tmp) != 0):
                        self.file_annotations.append(tmp)
                    tmp = []
                else:
                    values = line.split(",")
                    values[4] = values[4].rstrip("\n")
                    tmp.append(values)
            self.file_annotations.append(tmp)

    def save_dataset(self, name):
        X = np.array(self.X_data)
        np.save(name, X)
    def save_label(self, name):
        y = np.array(self.y_data)
        np.save(name, y)
    def open_dataset(self, name):
        self.X_data = np.load(name)
    def open_label(self, name):
        self.y_data = np.load(name)
        




if __name__ == '__main__':
    c = Classify()

    # # Cropped Version
    # folder_names = ["bird_vs_nonbird", "hawk_vs_crow"]
    # num_of_files = [5, 10]
    # c.read_all_cropped_image_list("../../data/Classify/cropped/image_lists", folder_names, num_of_files)
    # cv2.imshow("image", c.read_resize_image(c.train_cropped_images_filenames[15000]))
    # cv2.waitKey(0)



################################################



    # Real Dataset Version

    # c.open_dataset("data/Flying/x_data.npy")
    # c.open_label("data/Flying/y_data.npy")


    # print(c.X_data[10].shape)
    # print(c.y_data.shape)
    X1 = c.read_npy("data/Flying/x_data.npy")
    y1 = c.read_npy("data/Flying/y_data.npy")
    # X2 = c.read_npy("data/Flying/x_test3.npy")
    # y2 = c.read_npy("data/Flying/y_test3.npy")
    X2 = c.read_npy("data/Flying/x_train3.npy")
    y2 = c.read_npy("data/Flying/y_train3.npy")

    X = np.vstack((X1, X2))
    print("y1.shape = ", y1.shape)
    print("y2.shape = ", y2.shape)
    # print("y3.shape = ", y3.shape)

    print("x1.shape = ", X1.shape)
    print("x2.shape = ", X2.shape)
    # print("x3.shape = ", X3.shape)
    y = np.vstack((y1, y2))
    # y = y.reshape(len(y), 1)


    n, xs, ys, c = X.shape
    print("n = ", n, "xs = ", xs, "ys = ", ys, "c = ", c)
    X_new = np.zeros((20*n, xs, ys, c))
    y_new = np.zeros((20*n, 1))

    print("y.shape = ", y.shape)
    print("y_new.shape = ", y_new.shape)
    print("X.shape = ", X.shape)
    print("X_new.shape = ", X_new.shape)
    

    for i in range(n):
        # cv2.imshow("image", c.X_data[i])
        #plt.subplot2grid((4, 5), (0, 0))
        #plt.title("Original")
        #plt.imshow(X[i])
        X_new[20*i] = X[i]
        y_new[20*i] = y[i]

        hflipped = np.fliplr(X[i])
        #plt.subplot2grid((4, 5), (0, 1))
        #plt.title("horiz flipped")
        #plt.imshow(hflipped)
        X_new[20*i + 1] = hflipped
        y_new[20*i + 1] = y[i]


        vflipped = np.flipud(X[i])
        #plt.subplot2grid((4, 5), (0, 2))
        #plt.title("vert flipped")
        #plt.imshow(vflipped)
        X_new[20*i + 2] = vflipped
        y_new[20*i + 2] = y[i]


        r_im = tr.rotate(X[i], angle=30, mode="edge")
        #plt.subplot2grid((4, 5), (0, 3))
        #plt.title("rot 30")
        #plt.imshow(r_im)
        X_new[20*i + 3] = r_im
        y_new[20*i + 3] = y[i]



        r_im2 = tr.rotate(X[i], angle=60, mode="edge")
        #plt.subplot2grid((4, 5), (0, 4))
        #plt.title("rot 60")
        #plt.imshow(r_im2)
        X_new[20*i + 4] = r_im2
        y_new[20*i + 4] = y[i]



        r_im3 = tr.rotate(X[i], angle=120, mode="edge")
        #plt.subplot2grid((4, 5), (1, 0))
        #plt.title("rot 120")
        #plt.imshow(r_im3)
        X_new[20*i + 5] = r_im3
        y_new[20*i + 5] = y[i]



        r_im4 = tr.rotate(X[i], angle=150, mode="edge")
        #plt.subplot2grid((4, 5), (1, 1))
        #plt.title("rot 150")
        #plt.imshow(r_im4)
        X_new[20*i + 6] = r_im4
        y_new[20*i + 6] = y[i]



        r_im5 = tr.rotate(X[i], angle=-30, mode="edge")
        #plt.subplot2grid((4, 5), (1, 2))
        #plt.title("rot -30")
        #plt.imshow(r_im5)
        X_new[20*i + 7] = r_im5
        y_new[20*i + 7] = y[i]



        r_im6 = tr.rotate(X[i], angle=-60, mode="edge")
        #plt.subplot2grid((4, 5), (1, 3))
        #plt.title("rot -60")
        #plt.imshow(r_im6)
        X_new[20*i + 8] = r_im6
        y_new[20*i + 8] = y[i]



        r_im7 = tr.rotate(X[i], angle=-120, mode="edge")
        #plt.subplot2grid((4, 5), (1, 4))
        #plt.title("rot -120")
        #plt.imshow(r_im7)
        X_new[20*i + 9] = r_im7
        y_new[20*i + 9] = y[i]



        r_im8 = tr.rotate(X[i], angle=-150, mode="edge")
        #plt.subplot2grid((4, 5), (2, 0))
        #plt.title("rot -150")
        #plt.imshow(r_im8)
        X_new[20*i + 10] = r_im8
        y_new[20*i + 10] = y[i]



        shift_r = tr.AffineTransform(translation=(-5, 0))
        wr_im = tr.warp(X[i], shift_r, mode="edge")
        #plt.subplot2grid((4, 5), (2, 1))
        #plt.title("warp right x")
        #plt.imshow(wr_im)
        X_new[20*i + 11] = wr_im
        y_new[20*i + 11] = y[i]



        shift_l = tr.AffineTransform(translation=(5, 0))
        wl_im = tr.warp(X[i], shift_l, mode="edge")
        #plt.subplot2grid((4, 5), (2, 2))
        #plt.title("warp left x")
        #plt.imshow(wl_im)
        X_new[20*i + 12] = wl_im
        y_new[20*i + 12] = y[i]



        shift_rr = tr.AffineTransform(translation=(-5, -5))
        wrr_im = tr.warp(X[i], shift_rr, mode="edge")
        #plt.subplot2grid((4, 5), (2, 3))
        #plt.title("warp right xy")
        #plt.imshow(wrr_im)
        X_new[20*i + 13] = wrr_im
        y_new[20*i + 13] = y[i]



        shift_ll = tr.AffineTransform(translation=(5, 5))
        wll_im = tr.warp(X[i], shift_ll, mode="edge")
        #plt.subplot2grid((4, 5), (2, 4))
        #plt.title("warp left xy")
        #plt.imshow(wll_im)
        X_new[20*i + 14] = wll_im
        y_new[20*i + 14] = y[i]



        n_im = ut.random_noise(X[i], var=0.002)
        #plt.subplot2grid((4, 5), (3, 0))
        #plt.title("noisy")
        #plt.imshow(n_im)
        X_new[20*i + 15] = n_im
        y_new[20*i + 15] = y[i]



        b_im = cv2.GaussianBlur(X[i], (3, 3), 0)
        #plt.subplot2grid((4, 5), (3, 1))
        #plt.title("blurry")
        #plt.imshow(b_im)
        X_new[20*i + 16] = b_im
        y_new[20*i + 16] = y[i]



        contrast = 0.2
        cont_im = contrast*X[i]
        #plt.subplot2grid((4, 5), (3, 2))
        #plt.title("contrast 1")
        #plt.imshow(cont_im)
        X_new[20*i + 17] = cont_im
        y_new[20*i + 17] = y[i]



        contrast = 0.5
        cont_im1 = contrast*X[i]
        #plt.subplot2grid((4, 5), (3, 3))
        #plt.title("contrast 2")
        #plt.imshow(cont_im1)
        X_new[20*i + 18] = cont_im1
        y_new[20*i + 18] = y[i]



        contrast = 0.7
        cont_im2 = contrast*X[i]
        #plt.subplot2grid((4, 5), (3, 4))
        #plt.title("contrast 3")
        #plt.imshow(cont_im2)
        X_new[20*i + 19] = cont_im2
        y_new[20*i + 19] = y[i]






        


        #plt.pause(0.5)
        # print(y[i])
        if cv2.waitKey(0) > 0:
            continue
    # plt.show()

    # cv2.imshow("image", c.X_data[2])
    # for i in range(len(c.file_annotations)):
    #     print(str(c.file_annotations[i]) + "\n\n")
    # print(c.file_annotations)
    
    
    
    # for i in range(X_new.shape[0]):
    #     cv2.imshow("image", X_new[i])
    #     if cv2.waitKey(0) > 0:
    #         continue


    # if cv2.waitKey(0) > 0:
    #     pass

    np.save("X_data.npy", X_new)
    np.save("y_data.npy", y_new)