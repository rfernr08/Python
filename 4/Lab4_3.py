import matplotlib.pyplot as plt

# a. Ask the user on the screen the number of particles that will be on the system
num_particles = int(input("Insert the number of particles: "))

# Initialize lists to store the positions and masses
x_positions = []
y_positions = []
masses = []

# b. Ask the user on the screen the coordinates and the mass of each particle
for i in range(1, num_particles + 1):
    x = float(input(f"Particle {i}. Position x: "))
    y = float(input(f"Particle {i}. Position y: "))
    m = float(input(f"Particle {i}. Mass: "))
    x_positions.append(x)
    y_positions.append(y)
    masses.append(m)

# c. Calculate the coordinates of the center of mass
total_mass = sum(masses)
x_center_of_mass = sum(x*m for x, m in zip(x_positions, masses)) / total_mass
y_center_of_mass = sum(y*m for y, m in zip(y_positions, masses)) / total_mass

# d. Plot each particle and the center of mass
plt.scatter(x_positions, y_positions, color='blue')
for i in range(num_particles):
    plt.text(x_positions[i], y_positions[i], str(masses[i]))
plt.scatter(x_center_of_mass, y_center_of_mass, color='red', marker='^')
plt.text(x_center_of_mass, y_center_of_mass, str(total_mass))
plt.show()