import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)

    def fill_row(point_1, point_2): return np.array([[-point_2[0], -point_2[1], -1, 0, 0, 0, point_2[0]*point_1[0], point_2[1]*point_1[0], point_1[0]],
                                                     [0, 0, 0, -point_2[0], -point_2[1], -1, point_2[0]*point_1[1], point_2[1]*point_1[1], point_1[1]]])
    A = fill_row(p1[:, 0], p2[:, 0])
    for i in range(1, p1.shape[1]):
        A = np.concatenate((A, fill_row(p1[:, i], p2[:, i])), axis=0)
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    h = vh[8,:]
    H2to1 = np.array([[h[0], h[1], h[2]],
                      [h[3], h[4], h[5]],
                      [h[6], h[7], h[8]]])
    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    best_inliers = 0
    best_H = 0
    num_matches = matches.shape[0]
    locs1_matches = np.concatenate((np.transpose(locs1[matches[:, 0], 0:2]), np.ones((1, num_matches))), axis=0)
    locs2_matches = np.concatenate((np.transpose(locs2[matches[:, 1], 0:2]), np.ones((1, num_matches))), axis=0)
    for i in range(0, num_iter):
        random_four = np.random.randint(num_matches, size=4)
        p1 = locs1_matches[0:2, random_four]
        p2 = locs2_matches[0:2, random_four]
        sample_H = computeH(p1, p2)
        try:
            projected_points = np.matmul(np.linalg.inv(sample_H), locs1_matches)
        except:
            continue
        projected_points_norm = projected_points / projected_points[2,:]
        diff = np.sum(np.sqrt(np.square(locs2_matches-projected_points_norm)), axis=0)
        num_inliers = np.sum(diff < tol)
        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_H = sample_H

    projected_points = np.matmul(np.linalg.inv(best_H), locs1_matches)
    projected_points_norm = projected_points / projected_points[2, :]
    diff = np.sum(np.sqrt(np.square(locs2_matches - projected_points_norm)), axis=0)
    inliers = diff < tol

    bestH = computeH(locs1_matches[0:2, inliers], locs2_matches[0:2, inliers])


    return bestH
        
    

if __name__ == '__main__':
    # # My test
    # p1 = np.array([[1, 0, 3, 6, 84, 23, 58, 53, 21, 40, 11, 54, 32],
    #                [-5, 32, -99, 32, 1, 2, 3, 4, 5, 6, 7, 8, 32]])
    # p2 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, -10, -11, -14, -56],
    #                [-65, 43, 1, 9, 3, 32, 99, 4, 3, 2,1, 0, 4]])
    # p3 = np.array([[1, 2, 3, 4],
    #                [65, 43, 1, 9]])
    # p4 = np.array([[2, 4, 6, 8],
    #                [65, 43, 1, 9]])
    # H = computeH(p3, p4)
    # p1 = np.array([[1, 0, 3, 6, 84, 23, 58, 53, 21, 40, 11, 54, 32],
    #                [-5, 32, -99, 32, 1, 2, 3, 4, 5, 6, 7, 8, 32],
    #                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    # p2 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, -10, -11, -14, -56],
    #                [-65, 43, 1, 9, 3, 32, 99, 4, 3, 2, 1, 0, 4],
    #                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    # p3 = np.array([[1, 2, 3, 4],
    #                [65, 43, 1, 9],
    #                [1, 1, 1, 1]])
    # p4 = np.array([[2, 4, 6, 8],
    #                [65, 43, 1, 9],
    #                [1, 1, 1, 1]])
    # test1 = np.matmul(np.linalg.inv(H), p3)
    # test1 = test1 / test1[2, :]

    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    bestH = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)


