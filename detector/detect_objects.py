import cv2
import numpy as np
import scipy.ndimage.morphology
from planarH import ransacH
import matplotlib.patches
from BRIEF import briefLite,briefMatch,plotMatches
import matplotlib.pyplot as plt
import skimage.morphology
import skimage.segmentation





if __name__ == '__main__':
    image_folder = '/media/moon/Data/Brady/Video_9.avi-labels/'
    output_folder = '/media/moon/Data/Brady/Output/9/'
    save_full_output_images = False
    for j in range(1,1999):
        # path to image folder
        # im1 = cv2.imread('/media/moon/Data/Brady/Video_1_2.avi-labels/'+str(j)+'.png')
        # im2 = cv2.imread('/media/moon/Data/Brady/Video_1_2.avi-labels/'+str(j+1)+'.png')
        im1 = cv2.imread(image_folder+str(j)+'.png')
        im2 = cv2.imread(image_folder+str(j+1)+'.png')
        if np.any(im1) == None or np.any(im2) == None:
            continue

        # Find Homography
        locs1, desc1 = briefLite(im1)
        locs2, desc2 = briefLite(im2)
        matches = briefMatch(desc1, desc2)
        bestH = ransacH(matches, locs1, locs2, num_iter=5000, tol=.1)
        if len(im1.shape) > 2:
            h, w, _ = im1.shape
        else:
            h, w = im1.shape
        new_im2 = cv2.warpPerspective(im2, bestH, (w,h))
        # cv2.imshow('panoramas', new_im2)
        # cv2.imshow('im1', im1)

        # Background subtraction
        thresh = 7
        # thresh = .12
        diff = np.abs(new_im2[:,:,0].astype(float) - im1[:,:,0].astype(float))
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

        plt.imshow(im2, cmap='gray')
        for bbox in bboxes:
            minr, minc, maxr, maxc = bbox
            width = maxc - minc
            rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                                fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)

        if save_full_output_images:
            plt.savefig(output_folder + str(j + 1) + '.jpg')
            plt.clf()
        else:
            plt.show()








