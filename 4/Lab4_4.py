import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('iris.csv')

# Create a figure
fig = plt.figure()

# Create a 3D subplot
ax = fig.add_subplot(111, projection='3d')

# Create a dictionary to map varieties to shapes
markers = {'Setosa': 'o', 'Versicolor': 'v', 'Virginica': '^'}

# Plot each variety
for variety, marker in markers.items():
    variety_data = df[df['variety'] == variety]
    ax.scatter(variety_data['sepal_length'], variety_data['sepal_width'], variety_data['petal_length'], marker=marker)

# Set labels
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')

# Show the plot
plt.show()