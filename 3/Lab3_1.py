import numpy as np

# Create the following matrix and store it in the variable Matrix1
Matrix1 = np.array([[4, -2, 7], [9, 4, 1], [5, -1, 5]])

# Calculate its transposed and store it in the variable Matrix2
Matrix2 = Matrix1.T

# Calculate the element-wise product of Matrix1 and Matrix2
element_wise_product = Matrix1 * Matrix2

# Calculate the product of the matrices Matrix1 and Matrix2 and store it in the variable prodM1M2
prodM1M2 = Matrix1 @ Matrix2

# Calculate the product of the matrices Matrix2 and Matrix1 and store it in the variable prodM2M1
prodM2M1 = Matrix2 @ Matrix1

# Store in a 2D array called mat_corners the corners of Matrix1, writing a single line of code
mat_corners = Matrix1[[0, 0, -1, -1], [0, -1, 0, -1]].reshape(2, 2)

# Calculate the maximum of each row of Matrix1 and store it in vec_max. Also, calculate the global maximum of Matrix1
vec_max = Matrix1.max(axis=1)
global_max = Matrix1.max()

# Calculate the minimum of each column of Matrix1 and store it in vec_min. Also, calculate the global minimum of Matrix1
vec_min = Matrix1.min(axis=0)
global_min = Matrix1.min()

# Calculate the matrix product of vec_min and vec_max (in that order), so that the result is a matrix of shape (3, 3)
matrix_product = vec_min.reshape(-1, 1) @ vec_max.reshape(1, -1)

# Calculate the sum of the elements of the first and third column of Matrix1 and store them in a variable called mat_sum. Do it in only one line of code
mat_sum = Matrix1[:, [0, 2]].sum()