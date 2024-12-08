import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
# Generate random data points within the unit square

def set_bins_below_threshold_to_zero(bin_measurements, threshold):
    return np.where(bin_measurements < threshold, 0, bin_measurements)

def pret5(lstick, rstick, title):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    lstick[:, 1] = -lstick[:, 1]
    datasets = [lstick, rstick]
    titles = ['Left Joystick', 'Right Joystick']

    for i, data in enumerate(datasets):
        # Create a 2D histogram
        hist, xedges, yedges = np.histogram2d(data[:,0], data[:,1], bins=200, range=[[-1, 1], [-1, 1]])
        
        # Apply Gaussian smoothing with a higher sigma for a blob-like effect
        sigma = 3.0
        
        smoothed_hist = hist
        # smoothed_hist = set_bins_below_threshold_to_zero(smoothed_hist, 1e-8)
        print(np.min(smoothed_hist))
        # smoothed_hist = hist
        # Plot the smoothed heatmap with Spectral color map
        im = axs[i].imshow(smoothed_hist.T, origin='lower', extent=[-1, 1, -1, 1], cmap='Spectral', aspect='auto', norm=LogNorm())
        axs[i].add_artist(plt.Circle((0, 0), 1, color='red', fill=False, linewidth=2))
        axs[i].set_xlim(-1, 1)
        axs[i].set_ylim(-1, 1)
        axs[i].set_xlabel('x-axis')
        axs[i].set_ylabel('y-axis')
        axs[i].set_title(titles[i])
        
        # Add colorbar
        plt.colorbar(im, ax=axs[i], label='Density')

    plt.suptitle(title)
    plt.show()

# Call the function to display the plot