import numpy as np
from scipy.spatial import distance

# Generate a 10x4 matrix with random values between -10 and 10
matrix = np.random.uniform(-10, 10, (10, 4))

# a. Create a Euclidean distance matrix
dist_matrix = distance.cdist(matrix, matrix, 'euclidean')

print("Euclidean distance matrix:")
print(dist_matrix)

# b. Indicate the distance between the points, when such distance is lower than 10
rows, cols = np.where((dist_matrix > 0) & (dist_matrix < 10))
for i, j in zip(rows, cols):
    print(f"The Euclidean distance between vectors {i} and {j} is {dist_matrix[i, j]}")