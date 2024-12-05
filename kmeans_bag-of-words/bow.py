import numpy as np
import cv2
import glob
import os
from sklearn.cluster import KMeans
from tqdm import tqdm
import time

def findnn(D1, D2):
    """
    :param D1: NxD matrix containing N feature vectors of dim. D
    :param D2: MxD matrix containing M feature vectors of dim. D
    :return:
        Idx: N-dim. vector containing for each feature vector in D1 the index of the closest feature vector in D2.
        Dist: N-dim. vector containing for each feature vector in D1 the distance to the closest feature vector in D2
    """
    N = D1.shape[0]
    M = D2.shape[0]  # [k]

    # Find for each feature vector in D1 the nearest neighbor in D2
    Idx, Dist = [], []
    for i in range(N):
        minidx = 0
        mindist = np.linalg.norm(D1[i, :] - D2[0, :])
        for j in range(1, M):
            d = np.linalg.norm(D1[i, :] - D2[j, :])

            if d < mindist:
                mindist = d
                minidx = j
        Idx.append(minidx)
        Dist.append(mindist)
    return Idx, Dist


def grid_points(img, nPointsX, nPointsY, border):
    """
    :param img: input gray img, numpy array, [h, w]
    :param nPointsX: number of grids in x dimension
    :param nPointsY: number of grids in y dimension
    :param border: leave border pixels in each image dimension
    :return: vPoints: 2D grid point coordinates, numpy array, [nPointsX*nPointsY, 2]
    """
    vPoints = []  # numpy array, [nPointsX*nPointsY, 2]

    # Get the height and width of the image
    h, w = img.shape

    # Compute the spacing between points in the x and y dimensions
    stepX = (w - 2 * border) / (nPointsX - 1)
    stepY = (h - 2 * border) / (nPointsY - 1)

    for i in range(nPointsY):
        for j in range(nPointsX):
            x = border + j * stepX
            y = border + i * stepY
            vPoints.append([x, y])

    # Convert the list of points to a numpy array
    vPoints = np.array(vPoints)

    return vPoints


def descriptors_hog(img, vPoints, cellWidth, cellHeight):
    nBins = 8
    w = cellWidth
    h = cellHeight

    grad_x = cv2.Sobel(img, cv2.CV_16S, dx=1, dy=0, ksize=1)
    grad_y = cv2.Sobel(img, cv2.CV_16S, dx=0, dy=1, ksize=1)

    grad_x = np.float32(grad_x)
    grad_y = np.float32(grad_y)
    # Calculate magnitudes and angles
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    angle = np.arctan2(grad_y, grad_x) * (180 / np.pi)  # Convert to degrees
    angle[angle < 0] += 180  # Wrap angles to [0, 180]

    descriptors = []  # list of descriptors for the current image, each entry is one 128-d vector for a grid point
    for i in range(len(vPoints)):
        center_x = round(vPoints[i, 0])
        center_y = round(vPoints[i, 1])

        desc = np.zeros(nBins * 16)
        cell_count = 0
        for cell_y in range(-2, 2):
            for cell_x in range(-2, 2):
                start_y = center_y + (cell_y) * h
                end_y = center_y + (cell_y + 1) * h

                start_x = center_x + (cell_x) * w
                end_x = center_x + (cell_x + 1) * w

                # Ensure boundaries are within the image
                start_y = max(start_y, 0)
                end_y = min(end_y, img.shape[0])
                start_x = max(start_x, 0)
                end_x = min(end_x, img.shape[1])
                # Get the cells' magnitude and angle
                cell_magnitude = magnitude[start_y:end_y, start_x:end_x]
                # Normalize
                cell_magnitude = cell_magnitude / np.linalg.norm(cell_magnitude) if np.linalg.norm(cell_magnitude) != 0 else cell_magnitude
                cell_angle = angle[start_y:end_y, start_x:end_x]
                # Calculate the histogram for the cell
                hist = np.zeros(nBins)
                for j in range(cell_angle.shape[0]):
                    for k in range(cell_angle.shape[1]):
                        bin_index = int(cell_angle[j, k] // (180 / nBins)) % nBins
                        #bin_index = int(cell_angle[j, k] // (180 / nBins))  # Bin index
                        hist[bin_index] += cell_magnitude[j, k] # Accumulate magnitudes
                desc[cell_count * nBins:(cell_count + 1) * nBins] = hist
                cell_count += 1

                # compute the angles
                # compute the histogram

        descriptors.append(desc)

    # [nPointsX*nPointsY, 128], descriptor for the current image (100 grid points)
    descriptors = np.asarray(descriptors)
    return descriptors


def create_codebook(nameDirPos, nameDirNeg, k, numiter):
    """
    :param nameDirPos: dir to positive training images
    :param nameDirNeg: dir to negative training images
    :param k: number of kmeans cluster centers
    :param numiter: maximum iteration numbers for kmeans clustering
    :return: vCenters: center of kmeans clusters, numpy array, [k, 128]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDirPos, '*.png')))
    vImgNames = vImgNames + \
        sorted(glob.glob(os.path.join(nameDirNeg, '*.png')))

    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    # list for all features of all images (each feature: 128-d, 16 histograms containing 8 bins)
    vFeatures = []
    # Extract features for all image
    for i in tqdm(range(nImgs)):
        print('processing image {} ...'.format(i+1))
        img = cv2.imread(vImgNames[i])  # [h, w, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]

        # Collect local feature points for each image, and compute a descriptor for each local feature point
        vPoints = grid_points(img, nPointsX, nPointsY, border)
        # [100, 128]
        features = descriptors_hog(img, vPoints, cellWidth, cellHeight)
        # all features from one image [n_vPoints, 128] (100 grid points)
        vFeatures.append(features)

    vFeatures = np.asarray(vFeatures)  # [n_imgs, n_vPoints, 128]
    # [n_imgs*n_vPoints, 128]
    vFeatures = vFeatures.reshape(-1, vFeatures.shape[-1])
    print('number of extracted features: ', len(vFeatures))
    #print(np.isnan(vFeatures).any())
    # Cluster the features using K-Means
    print('clustering ...')
    start_time = time.time()  # Start timing
    kmeans_res = KMeans(n_clusters=k, max_iter=numiter).fit(vFeatures)
    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time  # Calculate elapsed time

    print(f'Clustering took {elapsed_time:.2f} seconds.')

    vCenters = kmeans_res.cluster_centers_  # [k, 128]

    return vCenters


def bow_histogram(vFeatures, vCenters):
    """
    :param vFeatures: MxD matrix containing M feature vectors of dim. D
    :param vCenters: NxD matrix containing N cluster centers of dim. D
    :return: histo: N-dim. numpy vector containing the resulting BoW activation histogram.
    """
    N = vCenters.shape[0]  # Number of cluster centers
    histo = np.zeros(N)  # Histogram of size N

    # Iterate through each feature vector
    for feature in vFeatures:
        # Calculate distances from the feature to all cluster centers
        distances = np.linalg.norm(vCenters - feature, axis=1)  # L2 distance

        # Find the index of the closest cluster center
        closest_center_index = np.argmin(distances)

        # Increment the histogram for the closest cluster center
        histo[closest_center_index] += 1
        # Apply L2 normalization
    histo /= np.sum(histo)


    return histo #[k,]


def create_bow_histograms(nameDir, vCenters):
    """
    :param nameDir: dir of input images
    :param vCenters: kmeans cluster centers, [k, 128] (k is the number of cluster centers)
    :return: vBoW: matrix, [n_imgs, k]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDir, '*.png')))
    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    # Extract features for all images in the given directory
    vBoW = []

    for i in tqdm(range(nImgs)):
        print('processing image {} ...'.format(i + 1))
        img = cv2.imread(vImgNames[i])  # [h, w, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]
        vPoints = grid_points(img, nPointsX, nPointsY, border)
        vFeatures = descriptors_hog(img, vPoints, cellWidth, cellHeight)
        histo = bow_histogram(vFeatures, vCenters)
        vBoW.append(histo)


    vBoW = np.asarray(vBoW)  # [n_imgs, k]

    return vBoW


def bow_recognition_nearest(histogram, vBoWPos, vBoWNeg):
    """
    :param histogram: bag-of-words histogram of a test image, [1, k]
    :param vBoWPos: bag-of-words histograms of positive training images, [n_imgs, k]
    :param vBoWNeg: bag-of-words histograms of negative training images, [n_imgs, k]
    :return: sLabel: predicted result of the test image, 0(without car)/1(with car)
    """
    DistPos = np.min(np.linalg.norm(vBoWPos - histogram, axis = 1))
    DistNeg = np.min(np.linalg.norm(vBoWNeg - histogram, axis = 1))
    # Find the nearest neighbor in the positive and negative sets and decide based on this neighbor
    if (DistPos < DistNeg):
        sLabel = 1
    else:
        sLabel = 0
    return sLabel

if __name__ == '__main__':
    # set a fixed random seed
    np.random.seed(42)
    
    nameDirPos_train = 'data/data_bow/cars-training-pos'
    nameDirNeg_train = 'data/data_bow/cars-training-neg'
    nameDirPos_test = 'data/data_bow/cars-testing-pos'
    nameDirNeg_test = 'data/data_bow/cars-testing-neg'

    # number of k-means clusters
    k = 6
    # maximum iteration numbers for k-means clustering
    numiter = 50 

    print('creating codebook ...')
    vCenters = create_codebook(nameDirPos_train, nameDirNeg_train, k, numiter)

    print('creating bow histograms (pos) ...')
    vBoWPos = create_bow_histograms(nameDirPos_train, vCenters)
    print('creating bow histograms (neg) ...')
    vBoWNeg = create_bow_histograms(nameDirNeg_train, vCenters)

    # test pos samples
    print('creating bow histograms for test set (pos) ...')
    vBoWPos_test = create_bow_histograms(
        nameDirPos_test, vCenters)  # [n_imgs, k]
    result_pos = 0
    print('testing pos samples ...')
    for i in range(vBoWPos_test.shape[0]):
        cur_label = bow_recognition_nearest(
            vBoWPos_test[i:(i+1)], vBoWPos, vBoWNeg)
        result_pos = result_pos + cur_label
    acc_pos = result_pos / vBoWPos_test.shape[0]
    print('test pos sample accuracy:', acc_pos)

    # test neg samples
    print('creating bow histograms for test set (neg) ...')
    vBoWNeg_test = create_bow_histograms(
        nameDirNeg_test, vCenters)  # [n_imgs, k]
    result_neg = 0
    print('testing neg samples ...')
    for i in range(vBoWNeg_test.shape[0]):
        cur_label = bow_recognition_nearest(
            vBoWNeg_test[i:(i + 1)], vBoWPos, vBoWNeg)
        result_neg = result_neg + cur_label
    acc_neg = 1 - result_neg / vBoWNeg_test.shape[0]
    print('test neg sample accuracy:', acc_neg)
#K=4,0.878,0.72,0.90s
#K=5,1.0,0.9,0.95s
##K=6,0.959,0.960,1.06s
#K=7,1,0.84,1.20s
#K=8,0.898,0.960,1.29s
#K=9,0.898,0.960,1.46s
#K=10,0.918,0.960,2.13s
#K=11,0.898,0.98,1.74s
#K=15,0.878,0.94,2.02s
#K=20,0.939,0.88,2.46s


