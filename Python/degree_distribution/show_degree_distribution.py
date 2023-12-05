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

    # Create a plot
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 9))

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
    ax1.plot(np.exp(centers), N, '-*', markersize=10, color=[0, 0, 1], linewidth=3)
    fit_curve = np.exp(fitfun(centers, b, alpha))
    ax1.plot(np.exp(centers), fit_curve, '-', linewidth=3, color='red', alpha=0.85)

    # Set plot properties
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title("General networks", fontsize=16)
    ax1.set_xlabel('k', fontsize=16)
    ax1.set_ylabel("CDF", fontsize=16)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)

    # Add formula to the plot
    formula = f'  y = {np.round(b,3)} -  {np.round(alpha,3)} k' + '\n' + r'$|G| = 10^{5}$'
    ax1.text(0.53, 0.89, formula, fontsize=18, transform=ax1.transAxes, bbox=dict(facecolor='grey', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.3))

    # Manually define position and size for the inset plot
    inset_x = 0.21
    inset_y = 0.16
    inset_width = 0.27
    inset_height = 0.26
    
    # Add inset plot
    ax_inset = fig.add_axes([inset_x, inset_y, inset_width, inset_height])

    # Plot fit on inset
    ax_inset.plot(np.exp(centers), fit_curve, '-', linewidth=3, color='red', alpha=0.6)
    ax_inset.plot(np.exp(centers), N, '-*', markersize=12, color=[0, 0, 1], linewidth=3)

    # Set inset properties
    ax_inset.set_title('')
    ax_inset.set_xlabel('')
    ax_inset.set_ylabel('')
    ax_inset.set_yscale('log')

    # Save the plot
    save_name = os.path.join(sys.path[0], 'figures/power_law_fit.pdf')
    plt.savefig(save_name, dpi=600, transparent=True, bbox_inches='tight')

    # Display the plot
    plt.show()