import cv2
import numpy as np
import scipy.ndimage.morphology
from planarH import ransacH
import matplotlib.patches
from BRIEF import briefLite, briefMatch, plotMatches
import matplotlib.pyplot as plt
import skimage.morphology
import skimage.segmentation


def find_moving_objects(im1, im2, save_full_output_images, output_folder, thresh=7):
    # Find Homography
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    if type(locs1) is bool or type(locs2) is bool:
        return False, False
    matches = briefMatch(desc1, desc2)
    bestH = ransacH(matches, locs1, locs2, num_iter=5000, tol=.1)
    if len(im1.shape) > 2:
        h, w, _ = im1.shape
    else:
        h, w = im1.shape
    new_im2 = cv2.warpPerspective(im2, bestH, (w, h))
    # cv2.imshow('panoramas', new_im2)
    # cv2.imshow('im1', im1)

    # Background subtraction
    # thresh = 7
    # thresh = .12
    diff = np.abs(new_im2[:, :, 0].astype(float) - im1[:, :, 0].astype(float))
    # cv2.imshow('diff', diff)
    mask = diff > thresh
    mask = scipy.ndimage.binary_erosion(mask)
    # mask = scipy.ndimage.binary_erosion(mask)
    mask = scipy.ndimage.binary_dilation(mask)
    mask = scipy.ndimage.binary_dilation(mask)

    masked = np.ma.masked_where(mask == False, mask)
    # fig, ax = plt.subplots(1, frameon=False)
    # ax.set_axis_off()
    # plt.imshow(im1, cmap='gray')
    # plt.imshow(masked, cmap='jet', alpha=1)
    # plt.show()
    # yo = 2

    # Make bounding boxes
    bboxes = []

    # remove connections to image border
    cleared = skimage.segmentation.clear_border(mask)

    # label regions
    label_image = skimage.measure.label(cleared, connectivity=1)

    for region in skimage.measure.regionprops(label_image):
        # keep regions with specific area?
        if region.bbox_area >= .1:
            minr, minc, maxr, maxc = region.bbox
            bboxes.append(region.bbox)

    bw = np.ones(mask.shape)
    bw[cleared] = 0

    plt.imshow(im1, cmap='gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        width = maxc - minc
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                            fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)

    if save_full_output_images:
        plt.savefig(output_folder + str(j) + '.jpg')
        plt.clf()
    else:
        plt.show()
    return new_im2, bboxes


def format_for_network(bboxes, new_im2):
    # Resize sample
    data = np.empty((len(bboxes),28,28,3))
    iter = 0
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        # cropped_box = np.copy(bw[minr:maxr + 1, minc:maxc + 1])
        height = maxr - minr
        width = maxc - minc
        even_pad = .1
        if height == 0 or width == 0:
            continue
        elif height > width:
            diff = height - width
            pad_value = int(np.floor(height * even_pad))
            pad_value_width = int(np.floor(diff / 2))
            pad_box = np.copy(new_im2[minr - pad_value:minr + height + pad_value,
                              minc - pad_value_width - pad_value:minc + width + pad_value_width + pad_value, :])
            # pad_box = np.pad(cropped_box, ((0, 0), (pad_value, pad_value)), 'constant', constant_values=1)
        elif height < width:
            diff = width - height
            pad_value = int(np.floor(width * even_pad))
            pad_value_height = int(np.floor(diff / 2))
            pad_box = np.copy(new_im2[minr - pad_value_height - pad_value:minr + height + pad_value_height + pad_value,
                              minc - pad_value:minc + width + pad_value])
        else:
            pad_value = int(np.floor(width * even_pad))
            pad_box = np.copy(new_im2[minr - pad_value:minr + height + pad_value,
                              minc - pad_value:minc + width + pad_value])
        if pad_box.shape[0] == 0 or pad_box.shape[1] == 0:
            continue
        resized_im = skimage.transform.resize(pad_box, (28, 28, 3), anti_aliasing=True)
        data[iter,:,:,:] = resized_im
        iter += 1
        # plt.imshow(resized_im, cmap='gray')
        # plt.show()
    return data


def filter_label(bboxes, im1_labels):
    if len(im1_labels.shape) > 1:
        label_num = im1_labels.shape[0]
    elif im1_labels.shape[0] == 0:
        label_num = 0
    else:
        label_num = 1
    for label in range(0, label_num):
        new_bboxes = []
        if label_num == 0:
            continue
        elif label_num == 1:
            minx_l = int(im1_labels[1])
            miny_l = int(im1_labels[2])
            maxx_l = int(im1_labels[3])
            maxy_l = int(im1_labels[4])
            for bbox in bboxes:
                miny, minx, maxy, maxx = bbox
                if (maxy > miny_l > miny and maxx > minx_l > minx) or \
                    (maxy > maxy_l > miny and maxx > minx_l > minx) or \
                    (maxy > miny_l > miny and maxx > maxx_l > minx) or \
                    (maxy > maxy_l > miny and maxx > maxx_l > minx) or \
                    (maxy_l > miny > miny_l and maxx_l > minx > minx_l) or \
                    (maxy_l > maxy > miny_l and maxx_l > minx > minx_l) or \
                    (maxy_l > miny > miny_l and maxx_l > maxx > minx_l) or \
                    (maxy_l > maxy > miny_l and maxx_l > maxx > minx_l):
                    continue
                else:
                    new_bboxes.append(bbox)
        else:
            minx_l = int(im1_labels[label, 1])
            miny_l = int(im1_labels[label, 2])
            maxx_l = int(im1_labels[label, 3])
            maxy_l = int(im1_labels[label, 4])
            for bbox in bboxes:
                miny, minx, maxy, maxx = bbox
                if (maxy > miny_l > miny and maxx > minx_l > minx) or \
                        (maxy > maxy_l > miny and maxx > minx_l > minx) or \
                        (maxy > miny_l > miny and maxx > maxx_l > minx) or \
                        (maxy > maxy_l > miny and maxx > maxx_l > minx) or \
                        (maxy_l > miny > miny_l and maxx_l > minx > minx_l) or \
                        (maxy_l > maxy > miny_l and maxx_l > minx > minx_l) or \
                        (maxy_l > miny > miny_l and maxx_l > maxx > minx_l) or \
                        (maxy_l > maxy > miny_l and maxx_l > maxx > minx_l):
                    continue
                else:
                    new_bboxes.append(bbox)

        bboxes = new_bboxes
    return bboxes


if __name__ == '__main__':
    image_folder = ['/media/moon/Data/Brady/Video_1_1.avi-labels/',
                    '/media/moon/Data/Brady/Video_1_2.avi-labels/',
                    '/media/moon/Data/Brady/Video_1_3.avi-labels/',
                    '/media/moon/Data/Brady/Video_3.avi-labels/',
                    '/media/moon/Data/Brady/Video_5.avi-labels/',
                    '/media/moon/Data/Brady/Video_6.avi-labels/',
                    '/media/moon/Data/Brady/Video_7.avi-labels/',
                    '/media/moon/Data/Brady/Video_9.avi-labels/',
                    '/media/moon/Data/Brady/Video_11.avi-labels/',
                    '/media/moon/Data/Brady/Video_13.avi-labels/',
                    '/media/moon/Data/Brady/Video_14.avi-labels/',
                    '/media/moon/Data/Brady/Video_16.avi-labels/',
                    '/media/moon/Data/Brady/Video_17.avi-labels/',
                    '/media/moon/Data/Brady/Video_21.avi-labels/',
                    '/media/moon/Data/Brady/Video_39.avi-labels/']
    output_folder = ['/media/moon/Data/Brady/Output/1_1/',
                     '/media/moon/Data/Brady/Output/1_2/',
                     '/media/moon/Data/Brady/Output/1_3/',
                     '/media/moon/Data/Brady/Output/3/',
                     '/media/moon/Data/Brady/Output/5/',
                     '/media/moon/Data/Brady/Output/6/',
                     '/media/moon/Data/Brady/Output/7/',
                     '/media/moon/Data/Brady/Output/9/',
                     '/media/moon/Data/Brady/Output/11/',
                     '/media/moon/Data/Brady/Output/13/',
                     '/media/moon/Data/Brady/Output/14/',
                     '/media/moon/Data/Brady/Output/16/',
                     '/media/moon/Data/Brady/Output/17/',
                     '/media/moon/Data/Brady/Output/21/',
                     '/media/moon/Data/Brady/Output/39/']
    save_full_output_images = True
    x_data = np.empty((0,28,28,3), dtype='uint8')
    y_data = np.empty((0,1), dtype='uint8')
    for folder_num in range(3, 15): #15
        for j in range(1, 3000):
            # path to image folder
            # im1 = cv2.imread('/media/moon/Data/Brady/Video_1_2.avi-labels/'+str(j)+'.png')
            # im2 = cv2.imread(/media/moon/Data/Brady/Video_1_2.avi-labels/'+str(j+1)+'.png')
            im1 = cv2.imread(image_folder[folder_num] + str(j) + '.png')
            im2 = cv2.imread(image_folder[folder_num] + str(j + 1) + '.png')
            if np.any(im1) == None or np.any(im2) == None:
                continue
            im1_label = np.genfromtxt(image_folder[folder_num] + str(j) + '.label', dtype='str')

            # plt.imshow(im1, cmap='gray')
            # minx_l = int(im1_label[1])
            # miny_l = int(im1_label[2])
            # maxx_l = int(im1_label[3])
            # maxy_l = int(im1_label[4])
            # rect = matplotlib.patches.Rectangle((minx_l, miny_l), maxx_l - minx_l, maxy_l - miny_l,
            #                                     fill=False, edgecolor='red', linewidth=2)
            # plt.gca().add_patch(rect)
            # plt.show()

            new_im2, bboxes = find_moving_objects(im1, im2, save_full_output_images, output_folder[folder_num])
            if type(new_im2) is bool:
                continue
            bboxes = filter_label(bboxes, im1_label)

            # plt.imshow(im1, cmap='gray')
            # for bbox in bboxes:
            #     minr, minc, maxr, maxc = bbox
            #     width = maxc - minc
            #     rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
            #                                         fill=False, edgecolor='red', linewidth=2)
            #     plt.gca().add_patch(rect)
            # plt.show()
            new_array_to_append = format_for_network(bboxes, new_im2)
            x_data = np.concatenate((x_data, new_array_to_append), axis=0)
            y_data = np.concatenate((y_data,np.zeros((new_array_to_append.shape[0],1))), axis=0)

        # Training and testing split
        indices = np.random.permutation(x_data.shape[0])
        split = int(np.floor(.8*x_data.shape[0]))
        training_idx, test_idx = indices[:split], indices[split:]
        np.save('x_train3',x_data[training_idx,:,:,:])
        np.save('y_train3', y_data[training_idx,:])
        np.save('x_test3', x_data[test_idx,:,:,:])
        np.save('y_test3', x_data[test_idx,:])
