import numpy as np

testList = [0, 1 ,2 , 4]
matrixList = np.array(testList)
print(matrixList)
# Generate a square matrix with 400 points
matrix = np.random.uniform(0, 3, (20, 20))

# The original matrix
print("Original matrix:")
print(matrix)


coords_b = np.where((matrix > 1) & (matrix < 2))
print("\nCoordinates of elements between 1 and 2:")
print(list(zip(*coords_b)))


coords_c = np.where((matrix < 1) | (matrix > 2))
print("\nCoordinates of elements lower than 1 or greater than 2:")
print(list(zip(*coords_c)))

# Round the generated matrix, and then print the coordinates of the values which are different than 1 in the rounded matrix
rounded_matrix = np.round(matrix)
coords_d = np.where(rounded_matrix != 1)
print("\nCoordinates of elements different than 1 in the rounded matrix:")
print(list(zip(*coords_d)))