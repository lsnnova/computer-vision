import numpy as np
import cv2
from scipy import signal #for the scipy.signal.convolve2d function
from scipy import ndimage #for the scipy.ndimage.maximum_filter

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0
    # 1. Compute image gradients in x and y direction
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
    dx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    dy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Ix = signal.convolve2d(img, dx, mode='same')
    Iy = signal.convolve2d(img, dy, mode='same')

    print("Hello World!")
    #raise NotImplementedError
    
    # 2. (Optional) Blur the computed gradients
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)
    b_Ix = cv2.GaussianBlur(Ix, ksize = (7, 7), sigmaX = sigma, borderType=cv2.BORDER_REPLICATE)
    b_Iy = cv2.GaussianBlur(Iy, ksize = (7, 7), sigmaX = sigma, borderType=cv2.BORDER_REPLICATE)

    #raise NotImplementedError

    # 3. Compute elements of the local auto-correlation matrix "M"

    Ix2 = b_Ix**2
    Iy2 = b_Iy**2
    Ixy = b_Ix * b_Iy

    #gaussian filtering
    Sxx = cv2.GaussianBlur(Ix2, (7, 7), sigma)
    Syy = cv2.GaussianBlur(Iy2, (7, 7), sigma)
    Sxy = cv2.GaussianBlur(Ixy, (7, 7), sigma)

    # You may refer to cv2.GaussianBlur or scipy.signal.convolve2d to perform the weighted sum

    #raise NotImplementedError

    # 4. Compute Harris response function C
    C = (Sxx * Syy - Sxy ** 2) - k * ((Sxx + Syy) ** 2)
    #raise NotImplementedError

    # 5. Detection with threshold and non-maximum suppression
    threshold = thresh * C.max()
    corner_candidates = np.zeros_like(C)
    corner_candidates[C > threshold] = 255

    C_local_max = ndimage.maximum_filter(C, size=5)  #5*5 window
    corners = (C == C_local_max) & (C > threshold)
    y_coords, x_coords = np.where(corners)
    corners = np.stack((x_coords, y_coords), axis=-1)
    #You may refer to np.stack to stack the coordinates to the correct output format
    #raise NotImplementedError
    return corners, C

