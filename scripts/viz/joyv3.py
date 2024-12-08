import matplotlib.pyplot as plt
import numpy as np

# Generate random data points within the unit square
np.random.seed(0)  # For reproducibility
x = np.random.uniform(-1, 1, 500)
y = np.random.uniform(-1, 1, 500)

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(x, y, alpha=0.5, label='Data points')

# Plot the unit circle
circle = plt.Circle((0, 0), 1, color='red', fill=False, linewidth=2, label='Unit circle')
ax.add_artist(circle)

# Set limits and labels
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_title('Scatter Plot with Unit Circle')

# Add grid and legend
ax.grid(True)
ax.legend()

# Show the plot
plt.show()