import numpy as np
import cv2

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    DoG_levels = levels[1:]
    for i in DoG_levels:
        DoG_pyramid.append(gaussian_pyramid[:,:,i+1]-gaussian_pyramid[:,:,i])
    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    DoG_shape = np.shape(DoG_pyramid)
    principal_curvature = np.empty(DoG_shape)
    for i in range(0,DoG_shape[2]):
        sobel_xx = cv2.Sobel(DoG_pyramid[:, :, i], dx=2, dy=0, ksize=3, ddepth=-1)
        sobel_xy = cv2.Sobel(DoG_pyramid[:, :, i], dx=1, dy=1, ksize=3, ddepth=-1)
        sobel_yx = sobel_xy
        sobel_yy = cv2.Sobel(DoG_pyramid[:, :, i], dx=0, dy=2, ksize=3, ddepth=-1)
        hessian1 = np.concatenate((np.reshape(sobel_xx, (-1, 1, 1)), np.reshape(sobel_xy, (-1, 1, 1))), axis=2)
        hessian2 = np.concatenate((np.reshape(sobel_yx, (-1, 1, 1)), np.reshape(sobel_yy, (-1, 1, 1))), axis=2)
        hessian = np.concatenate((hessian1,hessian2), axis=1)
        trace_sq = np.square(np.trace(hessian, axis1=1, axis2=2))
        determinate = np.linalg.det(hessian)
        index_zero = determinate == 0
        determinate[index_zero] = .000000000000000000001
        principal_curvature[:, :, i] = np.reshape(trace_sq/determinate, (DoG_shape[0:2]))


    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    DoG_shape = np.shape(DoG_pyramid)
    for i in DoG_levels:
        points = np.argwhere(np.logical_and(np.abs(DoG_pyramid[:, :, i]) > th_contrast, principal_curvature[:, :, i] < th_r))  # outputs row, column. Need to make x, y
        points_maxmin = []
        for one_point in points:
            # test if local extrema
            value_check = []
            if one_point[0] != 0:
                value_check.append(DoG_pyramid[one_point[0]-1,one_point[1],i])
                if one_point[1] != 0:
                    value_check.append(DoG_pyramid[one_point[0]-1, one_point[1]-1, i])
                    value_check.append(DoG_pyramid[one_point[0], one_point[1]-1, i])
                if one_point[1] != DoG_shape[1]-1:
                    value_check.append(DoG_pyramid[one_point[0]-1, one_point[1]+1, i])
                    value_check.append(DoG_pyramid[one_point[0], one_point[1]+1, i])
            if one_point[0] != DoG_shape[0]-1:
                value_check.append(DoG_pyramid[one_point[0]+1, one_point[1], i])
                if one_point[1] != 0:
                    value_check.append(DoG_pyramid[one_point[0]+1, one_point[1]-1, i])
                if one_point[1] != DoG_shape[1]-1:
                    value_check.append(DoG_pyramid[one_point[0]+1, one_point[1]+1, i])
            if i != 0:
                value_check.append(DoG_pyramid[one_point[0], one_point[1], i-1])
            if i != DoG_shape[2]-1:
                value_check.append(DoG_pyramid[one_point[0], one_point[1], i+1])
            neighbor_max = np.max(value_check)
            neighbor_min = np.min(value_check)
            current_value = DoG_pyramid[one_point[0], one_point[1], i]
            if current_value < neighbor_min:
                points_maxmin.append(one_point)
            elif current_value > neighbor_max:
                points_maxmin.append(one_point)
        if len(points_maxmin) < 1:
            return False
        points_extrema = np.stack(points_maxmin, axis=0)

        points_shape = np.shape(points_extrema)
        level = i*np.ones((points_shape[0]))
        if i == 0:
            locsDoG = np.stack([points_extrema[:, 1], points_extrema[:, 0], level], axis=1)
        else:
            locsDoG = np.concatenate((locsDoG, np.stack([points_extrema[:, 1], points_extrema[:, 0], level], axis=1)), axis=0)
    locsDoG = locsDoG.astype(int)
    return locsDoG

    

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    gauss_pyramid = createGaussianPyramid(im, sigma0, k, levels)
    DoG_pyramid, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    principal_curvature = computePrincipalCurvature(DoG_pyramid)
    locsDoG = getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature, th_contrast, th_r)

    return locsDoG, gauss_pyramid

def displayLocsDoG(img, locsDoG):
    '''
    Displays image with location of points.

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    Outputs         Description
    --------------------------------------------------------------------------

    '''
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.max()>10:
        img = np.float32(img)/255
    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    scalar = 10 #scales the image size
    img = cv2.resize(img, None, fx=scalar, fy=scalar)
    for point in locsDoG:
        cv2.circle(img, (int(point[0]*scalar), int(point[1]*scalar)), 10, (0, 1, 0), -1)
    cv2.namedWindow('Image with Keypoints', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Image with Keypoints', img)
    # Uncomment block below for saving the image
    # img = 255 * img
    # img = img.astype(np.uint8)
    # cv2.imwrite('../figures/keypoints.jpg', img)
    cv2.waitKey(0)  # press any key to exit
    cv2.destroyAllWindows()





if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)
    displayLocsDoG(im, locsDoG)




