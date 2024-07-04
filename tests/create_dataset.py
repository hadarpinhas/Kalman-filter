import numpy as np
import matplotlib.pyplot as plt
import json

def create_spiral_points(initial_radius, radius_increment, num_points, num_turns):
    angles = np.linspace(0, 2 * np.pi * num_turns, num_points)
    radius = initial_radius + radius_increment * angles / (2 * np.pi)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return x, y

def generate_drone_data(num_points, num_turns, initial_radius, radius_increment, initial_velocity, outlier_frequency):
    x, y = create_spiral_points(initial_radius, radius_increment, num_points, num_turns)
    vx = -initial_velocity * np.sin(np.linspace(0, 2 * np.pi * num_turns, num_points))
    vy = initial_velocity * np.cos(np.linspace(0, 2 * np.pi * num_turns, num_points))

    # Add randomness to positions and velocities
    position_noise = np.random.normal(0, 10, (num_points, 2))  # Adding noise with a standard deviation of 10
    velocity_noise = np.random.normal(0, 5, (num_points, 2))   # Adding noise with a standard deviation of 5

    x += position_noise[:, 0]
    y += position_noise[:, 1]
    vx += velocity_noise[:, 0]
    vy += velocity_noise[:, 1]

    measurements = np.column_stack((x, y, vx, vy))

    # Create control inputs and add randomness
    control_noise = np.random.normal(0, 0.5, (num_points, 2))  # Adding noise with a standard deviation of 0.5
    control_inputs = np.array([[2, 2]] * num_points) + control_noise

    # Add outliers
    for i in range(0, num_points, outlier_frequency):
        measurements[i, :2] += np.random.normal(100, 50, 2)  # Large outlier noise in position
        measurements[i, 2:] += np.random.normal(20, 10, 2)   # Large outlier noise in velocity

    return measurements, control_inputs

num_points = 300  # More points for a smoother spiral
num_turns = 3  # Number of turns in the spiral
initial_radius = 500
radius_increment = 500
initial_velocity = 50
outlier_frequency = 20  # Add an outlier every 20 points

measurements, control_inputs = generate_drone_data(num_points, num_turns, initial_radius, radius_increment, initial_velocity, outlier_frequency)

# Create a list of dictionaries for JSON
data_list = [
    {
        'x_position': measurements[i, 0],
        'y_position': measurements[i, 1],
        'x_velocity': measurements[i, 2],
        'y_velocity': measurements[i, 3],
        'control_input_x': control_inputs[i, 0],
        'control_input_y': control_inputs[i, 1],
    }
    for i in range(num_points)
]

# Save list of dictionaries to a JSON file
with open('measurements.json', 'w') as f:
    json.dump(data_list, f)

# Plotting the generated points
plt.figure(figsize=(10, 10))

# Plot the spiral path as a dashed line
plt.plot(measurements[:, 0], measurements[:, 1], 'r--', label='Spiral Path')

# Plot the measured points as dots
plt.scatter(measurements[:, 0], measurements[:, 1], label='Measurements', color='blue', s=10)

# Highlight outliers
outliers = measurements[::outlier_frequency]
plt.scatter(outliers[:, 0], outliers[:, 1], label='Outliers', color='red', s=30)

# Sparse quiver plot for velocities
plt.quiver(measurements[::10, 0], measurements[::10, 1], 
           measurements[::10, 2], measurements[::10, 3])

plt.xlabel('x position [m]')
plt.ylabel('y position [m]')
plt.legend()
plt.title('Drone Position and Velocity in a Spiral Path with Randomness and Outliers')
plt.grid(True)
plt.show()
