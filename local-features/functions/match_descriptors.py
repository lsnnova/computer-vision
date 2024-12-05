import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = np.sum((desc1[:, np.newaxis, :] - desc2[np.newaxis, :, :]) ** 2, axis=2)

    return distances
    #raise NotImplementedError

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = []
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # You may refer to np.argmin to find the index of the minimum over any axis
        y_min = np.argmin(distances, axis=1)
        matches = list(enumerate(y_min))
        #raise NotImplementedError
    elif method == "mutual":
        # You may refer to np.min to find the minimum over any axis

        y_min = np.argmin(distances, axis=1)
        x_min = np.argmin(distances, axis=0)

        mutual_matches = np.where(np.arange(q1) == x_min[y_min])[0]
        matches = [(i, y_min[i]) for i in mutual_matches]

        #raise NotImplementedError
    elif method == "ratio":
        # You may use np.partition(distances,2,axis=0)[:,1] to find the second smallest value over a row
        #raise NotImplementedError
        min_two = np.partition(distances, 2, axis=1)[:, :2]
        min_val = min_two[:, 0]
        second_min_val = min_two[:, 1]

        ratio_condition = (min_val / second_min_val) < ratio_thresh
        y_min = np.argmin(distances, axis=1)
        matches = [(i, y_min[i]) for i in range(q1) if ratio_condition[i]]
    else:
        print("Please enter one method")
        #raise NotImplementedError
    return np.array(matches)

