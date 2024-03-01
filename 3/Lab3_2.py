import numpy as np

# Generate a square matrix with 400 points
matrix = np.random.uniform(0, 3, (20, 20))

# a. The original matrix
print("Original matrix:")
print(matrix)

# b. The coordinates of the elements whose value is between 1 and 2
coords_b = np.where((matrix > 1) & (matrix < 2))
print("\nCoordinates of elements between 1 and 2:")
print(list(zip(*coords_b)))

# c. The coordinates of the elements which are lower than 1 or greater than 2
coords_c = np.where((matrix < 1) | (matrix > 2))
print("\nCoordinates of elements lower than 1 or greater than 2:")
print(list(zip(*coords_c)))

# d. Round the generated matrix, and then print the coordinates of the values which are different than 1 in the rounded matrix
rounded_matrix = np.round(matrix)
coords_d = np.where(rounded_matrix != 1)
print("\nCoordinates of elements different than 1 in the rounded matrix:")
print(list(zip(*coords_d)))