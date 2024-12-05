import time
import os
import random
import math
import torch
import numpy as np

# run `pip install scikit-image==0.18.3` to install skimage, if you haven't done so.
# If you use scikit-image>=0.19, you will need to replace the `multichannel=True` argument with `channel_axis=-1`
# for the `skimage.transform.rescale` function
from skimage import io, color
from skimage.transform import rescale

def distance(x, X):
    dist = torch.sqrt(torch.sum((X - x) ** 2, dim=1))

    return dist

    #raise NotImplementedError('distance function not implemented!')

def distance_batch(X):
    dist = torch.cdist(X, X)

    return dist
    #raise NotImplementedError('distance_batch function not implemented!')

def gaussian(dist, bandwidth):
    #raise NotImplementedError('gaussian function not implemented!')
    weight = torch.exp(-0.5 * (dist / bandwidth) ** 2)

    return weight

def update_point(weight, X):
    #raise NotImplementedError('update_point function not implemented!')
    weighted_sum = torch.sum(weight.unsqueeze(1) * X, dim=0)
    total_weight = torch.sum(weight)
    new_point = weighted_sum / total_weight
    return new_point

def update_point_batch(weight, X):
    #raise NotImplementedError('update_point_batch function not implemented!')

    weighted_sum = torch.matmul(weight, X)
    total_weight = torch.sum(weight, dim=1, keepdim=True)
    total_weight = torch.clamp(total_weight, min=1e-10)

    new_points = weighted_sum / total_weight
    return new_points

def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    #raise NotImplementedError('meanshift_step_batch function not implemented!')
    dist = distance_batch(X)

    weights = gaussian(dist, bandwidth)

    new_X = update_point_batch(weights, X)
    return new_X

def meanshift(X):
    X = X.clone()
    for _ in range(20):
        #X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X)   # fast implementation
    return X

scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
#X = meanshift(torch.from_numpy(image_lab).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))
#22.7252956867218
#4.081002950668335

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
