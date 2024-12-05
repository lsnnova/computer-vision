import numpy as np
import cv2
from matplotlib import pyplot as plt
import time


# set a fixed random seed
np.random.seed(42)

# Load the image
image = cv2.imread('data/data_kmeans/img.jpg')  # Provide your image path
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image into a 2D array of pixels
pixel_values = image_rgb.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Define a function to calculate the Euclidean distance between two points
def euclidean_distance(pixel_values, centroids):

    return np.linalg.norm(pixel_values - centroids, axis=1)


# Define the K-means algorithm
def kmeans(pixel_values, centroids, max_iterations=50):
    iteration = 0
    for i in range(max_iterations):
        iteration += 1
        labels = np.zeros(pixel_values.shape[0], dtype=np.int32)
        for idx, pixel in enumerate(pixel_values):

            distances = euclidean_distance(pixel, centroids)
            labels[idx] = np.argmin(distances)

        # Recalculate the centroids
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            cluster_points = pixel_values[labels == k]
            new_centroids[k] = np.mean(cluster_points,axis=0)

        # Check for convergence
        # Stop when centroids no longer change
        if np.linalg.norm(centroids - new_centroids) < 1e-5:
            break
        centroids = new_centroids

    return labels, centroids, iteration

# Measure computational cost and iterations for different values of K
K_values = [2, 3, 5, 7, 10]  # Different values of K to test
time_costs = []  # List to store computation times
iterations_list = []  # List to store the number of iterations

for K in K_values:
    # Initial centroids
    centroids = pixel_values[np.random.choice(pixel_values.shape[0], K, replace=False)]

    start_time = time.time()
    labels, centroids, iterations = kmeans(pixel_values, centroids)
    elapsed_time = time.time() - start_time

    time_costs.append(elapsed_time)
    iterations_list.append(iterations)

    print(f'K = {K}, Time taken: {elapsed_time:.4f} seconds, Iterations: {iterations}')
    # Map each pixel to a distinct color based on its cluster label
    # Define a set of distinct colors for each cluster
    colors = np.array([
        [255, 0, 0],  # Red
        [0, 255, 0],  # Green
        [0, 0, 255],  # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [128, 0, 128],  # Purple
        [255, 165, 0],  # Orange
        [128, 128, 128],  # Gray
        [0, 128, 128],  # Teal
    ])

    # Ensure we only have as many colors as there are clusters
    colors = colors[:K]

    # Assign colors to each pixel based on the cluster it belongs to
    segmented_image = colors[labels]
    segmented_image = segmented_image.reshape(image_rgb.shape)

    # Show the original and segmented image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title(f'Segmented Image with K={K}')
    plt.show()


