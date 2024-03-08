import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the iris dataset into a pandas DataFrame
df = pd.read_csv('iris.csv')

# Create a dictionary to map the flower varieties to shapes
shapes = {'setosa': 'o', 'versicolor': 's', 'virginica': '^'}

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for variety, shape in shapes.items():
    subset = df[df['variety'] == variety]
    ax.scatter(subset['sepal.length'], subset['sepal.width'], subset['petal.length'], marker=shape)

ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
plt.show()