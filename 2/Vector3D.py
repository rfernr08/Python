import math
import pickle

class Vector3D:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def set_coordinates(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def add(self, other):
        self.x += other.x
        self.y += other.y
        self.z += other.z

    def subtract(self, other):
        self.x -= other.x
        self.y -= other.y
        self.z -= other.z

    def multiply(self, scalar):
        self.x *= scalar
        self.y *= scalar
        self.z *= scalar

    def modulus(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def save_to_txt(self, filename):
        with open(filename, 'w') as f:
            f.write(str(self))

    def save_to_pickle(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)