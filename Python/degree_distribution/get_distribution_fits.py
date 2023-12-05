import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import sys

def fitfun(x, b, alpha_cdf):
    """
    Return the logarithm of the power-law function used for curve fitting.

    Parameters:
    - x (numpy.ndarray): Input values.
    - b (float): Scale parameter.
    - alpha_cdf (float): Exponent parameter.

    Returns:
    - numpy.ndarray: Logarithm of the power-law function values.
    """
    return np.log(b) - alpha_cdf * x

def get_fit_parameters(x, y):
    """
    Perform curve fitting to obtain parameters for the power-law function.

    Parameters:
    - x (numpy.ndarray): Input values.
    - y (numpy.ndarray): Output values.

    Returns:
    - Tuple: Fit parameters (b, alpha).
    """
    popt, _ = curve_fit(fitfun, x, np.log(y), bounds=([0, 0], [np.inf, np.inf]), p0=[1, 1])

    b = popt[0]
    alpha = popt[1]

    return b, alpha

if __name__ == "__main__":
    # Load data
    load_name = os.path.join(sys.path[0], 'data/degrees.npy')
    degrees = np.load(load_name, allow_pickle=True)
    degrees = sorted(degrees)[2:]

    # Calculate the histogram
    N, edges = np.histogram(np.log(degrees), bins=35)

    # Convert to CCDF
    N = np.array([np.sum(N[i:]) for i in range(len(N))])

    # Remove y=0 probabilities
    nonzero_indices = np.where(N != 0)
    N = N[nonzero_indices]
    edges = edges[:-1][nonzero_indices]

    # Calculate bin centers
    centers = (edges[:-1] + edges[1:]) / 2

    # Normalize probabilities
    N = N[:-1]
    N = N / np.sum(N)

    # Get fit parameters
    b, alpha = get_fit_parameters(centers, N)

    # Plot the fit on top of the data
    fit_curve = np.exp(fitfun(centers, b, alpha))

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16, 9))
    fig.set_tight_layout(True)

    # Plot original data and fit
    ax1.plot(np.exp(centers), N, '-*', markersize=10, color=[0, 0, 1], linewidth=3)
    ax1.plot(np.exp(centers), fit_curve, '-', linewidth=3, color='red', alpha=0.85)
    ax1.text(0.4, 0.9, f'y = {np.round(b, 3)} - {np.round(alpha,3)} k', fontsize=16, transform=ax1.transAxes, color='red')

    # Compare with a different line (exponent = 2)
    fitting_function_2 = lambda x, b: np.log(b) - 2 * x
    [b_2], _ = curve_fit(fitting_function_2, centers, np.log(N), bounds=([0], [np.inf]), p0=[1])
    ax2.plot(np.exp(centers), N, '-*', markersize=10, color=[0, 0, 1], linewidth=3)
    ax2.plot(np.exp(centers), np.exp(fitfun(centers, b_2, 2)), '-', linewidth=3, color='green', alpha=0.85)
    ax2.text(0.4, 0.9, f'y = {np.round(b_2, 3)} - 2 k', fontsize=16, transform=ax2.transAxes, color='green')

    # Compare with a different line (exponent = 1.5)
    fitting_function_3 = lambda x, b: np.log(b) - 3/2 * x
    [b_3], _ = curve_fit(fitting_function_3, centers, np.log(N), bounds=([0], [np.inf]), p0=[1])  
    ax3.plot(np.exp(centers), N, '-*', markersize=10, color=[0, 0, 1], linewidth=3)
    ax3.plot(np.exp(centers), np.exp(fitfun(centers, b_3, 1.5)), '-', linewidth=3, color='#FFA500', alpha=0.85)
    ax3.text(0.4, 0.9, f'y = {np.round(b_3, 3)} - 1.5 k', fontsize=16, transform=ax3.transAxes, color='#FFA500')

    # Set properties for each subplot
    for ax in [ax1, ax2, ax3]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title("General networks", fontsize=16)
        ax.set_xlabel('k', fontsize=16)
        ax.set_ylabel("CDF", fontsize=16)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

    # Save the plot
    save_name = os.path.join(sys.path[0], 'figures/power_law_fit_separately.pdf')
    plt.savefig(save_name, dpi=600, transparent=True, bbox_inches='tight')

    # Display the plot
    plt.show()