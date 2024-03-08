import matplotlib.pyplot as plt
import numpy as np

# Create an array of angles
a = np.linspace(0, 2*np.pi, 100)

# Calculate the trigonometric functions
sin_a = np.sin(a)
cos_a = np.cos(a)
tan_a = np.tan(a)

sin_a_plus = np.sin(a + np.pi/4)
cos_a_plus = np.cos(a + np.pi/4)
tan_a_plus = np.tan(a + np.pi/4)

# Create a figure with 3 subplots
fig, axs = plt.subplots(3)

# Plot the sine functions
axs[0].plot(a, sin_a, color='red', label='sin(a)')
axs[0].plot(a, sin_a_plus, color='black', linestyle='dashed', label='sin(a + pi/4)')
axs[0].legend()

# Plot the cosine functions
axs[1].plot(a, cos_a, color='red', label='cos(a)')
axs[1].plot(a, cos_a_plus, color='black', linestyle='dashed', label='cos(a + pi/4)')
axs[1].legend()

# Plot the tangent functions
axs[2].plot(a, tan_a, color='red', label='tan(a)')
axs[2].plot(a, tan_a_plus, color='black', linestyle='dashed', label='tan(a + pi/4)')
axs[2].legend()

# Show the plot
plt.show()