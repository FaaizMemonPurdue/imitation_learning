import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
# Generate random data points within the unit square

def set_bins_below_threshold_to_zero(bin_measurements, threshold):
    return np.where(bin_measurements < threshold, 0, bin_measurements)

def pret(lstick, rstick, title):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    lstick[:, 1] = -lstick[:, 1]
    datasets = [lstick, rstick]
    titles = ['Left Joystick', 'Right Joystick']

    for i, data in enumerate(datasets):
        # Create a 2D histogram
        hist, xedges, yedges = np.histogram2d(data[:,0], data[:,1], bins=200, range=[[-1, 1], [-1, 1]])
        
        # Apply Gaussian smoothing with a higher sigma for a blob-like effect
        sigma = 3.0
        
        smoothed_hist = gaussian_filter(hist, sigma)
        # smoothed_hist = hist
        # Plot the smoothed heatmap with Spectral color map
        im = axs[i].imshow(smoothed_hist.T, origin='lower', extent=[-1, 1, -1, 1], cmap='Spectral', aspect='equal')
        axs[i].add_artist(plt.Circle((0, 0), 1, color='red', fill=False, linewidth=2))
        axs[i].set_xlim(-1, 1)
        axs[i].set_ylim(-1, 1)
        # axs[i].set_title(titles[i])
        
        # Add colorbar
        # plt.colorbar(im, ax=axs[i], label='Density')

    # plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def pret4s(lsticks, rsticks, title):
    fig, axs = plt.subplots(4, 2, figsize=(16, 32))
    titles = ['Left Joystick', 'Right Joystick']
    
    for i in range(4):
        lstick = lsticks[i]
        rstick = rsticks[i]
        
        lstick[:, 1] = -lstick[:, 1]
        datasets = [lstick, rstick]
        
        for j, data in enumerate(datasets):
            ax = axs[i, j]
            hist, xedges, yedges = np.histogram2d(data[:,0], data[:,1], bins=200, range=[[-1, 1], [-1, 1]])
            sigma = 3.0
            smoothed_hist = gaussian_filter(hist, sigma)
            im = ax.imshow(smoothed_hist.T, origin='lower', extent=[-1, 1, -1, 1], cmap='Spectral', aspect='equal')
            ax.add_artist(plt.Circle((0, 0), 1, color='red', fill=False, linewidth=2))
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_title(titles[j])
            plt.colorbar(im, ax=ax, label='Density')
    
    plt.suptitle(title)
    plt.show()
# Call the function to display the plot