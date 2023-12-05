import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
    # Load lengths
    load_name = os.path.join(sys.path[0], 'data/lengths.npy')
    lengths = np.load(load_name, allow_pickle=True)
    lengths = dict(lengths)
    lengths = np.array(list(lengths.values()))

    # Calculate the histogram
    N, edges = np.histogram(np.log(lengths), bins=35)

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

    # Create a plot
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 7))
    fig.set_tight_layout(True)

    # Plot original data and fit
    ax1.plot(np.exp(centers), N, '-*', markersize=10, color=[0, 0, 1], linewidth=3)
    ax1.plot(np.exp(centers), fit_curve, '-', linewidth=3, color='red', alpha=0.85)

    # Set plot properties
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title("Lengths Distribution", fontsize=16)
    ax1.set_xlabel('Length', fontsize=16)
    ax1.set_ylabel("CCDF", fontsize=16)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)

    # Save the plot (optional)
    save_name = os.path.join(sys.path[0], 'figures/lengths_distribution_fit.pdf')
    plt.savefig(save_name, dpi=600, transparent=True, bbox_inches='tight')

    # Display the plot
    plt.show()