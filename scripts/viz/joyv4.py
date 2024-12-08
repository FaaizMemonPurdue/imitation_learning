import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# Generate random data points within the unit square
np.random.seed(0)  # For reproducibility
def pret4(data, flip_y=True):
    # Create a 2D histogram
    if flip_y: data[:, 1] = -data[:, 1] 
    hist, xedges, yedges = np.histogram2d(data[:,0], data[:,1], bins=60, range=[[-1, 1], [-1, 1]])

    # Apply Gaussian smoothing
    sigma = 5.0
    smoothed_hist = gaussian_filter(hist, sigma)
    # smoothed_hist = hist
    # Plot the smoothed heatmap with Spectral color map
    plt.figure(figsize=(8, 8))
    plt.imshow(smoothed_hist.T, origin='lower', extent=[-1, 1, -1, 1], cmap='Spectral', aspect='auto')
    plt.colorbar(label='Density')

    # Plot the unit circle
    circle = plt.Circle((0, 0), 1, color='red', fill=False, linewidth=2)
    plt.gca().add_artist(circle)

    # Set limits and labels
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('Smoothed Heatmap with Spectral Color Map and Unit Circle')

    # Show the plot
    plt.show()
