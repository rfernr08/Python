from vector3d import Vector3D

# a. Initialize the vector with the coordinates (0, 0, 0)
v = Vector3D(0, 0, 0)
print(v)

# b. Change the values of the vector to (-6, 10, 5)
v.set_coordinates(-6, 10, 5)
print(v)

# c. Add the vector to another vector with coordinates (5, -1, 0)
v2 = Vector3D(5, -1, 0)
v.add(v2)
print(v)

# d. Subtract (-1, -1, -9) to the vector.
v3 = Vector3D(-1, -1, -9)
v.subtract(v3)
print(v)

# e. Multiply the resulting vector by 3.5.
v.multiply(3.5)
print(v)

# f. Return the modulus of the resulting vector.
print(v.modulus())

# g. Save the resulting vector on a txt and a pickle file.
v.save_to_txt('vector.txt')
v.save_to_pickle('vector.pkl')