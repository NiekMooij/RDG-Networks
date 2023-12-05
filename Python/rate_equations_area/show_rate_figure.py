import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy import stats

if __name__ == "__main__":
    # Load data
    load_name = os.path.join(sys.path[0], 'data/areas.npy')
    areas = np.load(load_name, allow_pickle=True)
    areas = dict(areas)

    load_name = os.path.join(sys.path[0], 'data/lengths.npy')
    lengths = np.load(load_name, allow_pickle=True)
    lengths = dict(lengths)

    load_name = os.path.join(sys.path[0], 'data/probabilities.npy')
    probabilities = np.load(load_name, allow_pickle=True)
    probabilities = dict(probabilities)

    # Display network size
    print(f'|G| = {len(lengths)}')

    # Create a scatter plot
    fig, ax1 = plt.subplots(figsize=(8, 9))
    
    # Combine data into a dictionary
    data = {id: [areas[id], lengths[id], probabilities[id]] for id in areas.keys()}    
    # Sort data by areas
    data = dict(sorted(data.items(), key=lambda item: item[1][0]))
    
    x = np.array([data[id][0] for id in data.keys()])  # Areas
    y = np.array([data[id][2] for id in data.keys()])  # Probabilities

    # Exclude points in the high area range
    points_to_neglect = 50
    keys_to_remove = list(data.keys())[-points_to_neglect:]
    for key in keys_to_remove:
        del data[key]
        
    x_sorted = np.array([data[id][0] for id in data.keys()])
    y_sorted = np.array([data[id][2] for id in data.keys()])

    # Scatter plot for the main plot
    ax1.scatter(x, y, s=20, marker='o', color='blue', label='Data Points', alpha=0.15)

    # Linear regression line with custom style
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    ax1.plot(x, intercept + slope * x, color='red', linestyle='--', label=f"Regression Line: y = {slope:.3f}x + {intercept:.3f}")

    # Add correspondence lines
    ax1.plot(x, intercept + 0.5 * x, color='green', linestyle='--', label=f"Line: y = {0.5:.3f}x + {intercept:.3f}")
    ax1.plot(x, intercept + 0.4 * x, color='#FFA500', linestyle='--', label=f"Line: y = {0.4:.3f}x + {intercept:.3f}")

    # Include R-squared value
    ax1.text(0.94, 0.07, r'$R^2$' + f': {r_value**2:.3f}', transform=ax1.transAxes, fontsize=16, horizontalalignment='right')
    ax1.text(0.94, 0.04, f'|G|: {len(lengths)-4}', transform=ax1.transAxes, fontsize=16, horizontalalignment='right')
    ax1.text(0.94, 0.01, f'Trials: 10000', transform=ax1.transAxes, fontsize=16, horizontalalignment='right')

    ax1.set_xlabel(r'$\sum A_i$', fontsize=16)
    ax1.set_ylabel(r'$P_{k\mapsto k+1}$', fontsize=16)
    ax1.set_title('General networks', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    ax1.legend(fontsize=16)

    # Manually define position and size for the inset plot
    inset_x = 0.143
    inset_y = 0.442
    inset_width = 0.27
    inset_height = 0.25

    ax_inset = fig.add_axes([inset_x, inset_y, inset_width, inset_height])
    ax_inset.scatter(x_sorted, y_sorted, s=20, marker='o', color='blue', label='Data Points', alpha=0.15)

    slope_sorted, intercept_sorted, r_value_sorted, p_value_sorted, std_err_sorted = stats.linregress(x_sorted, y_sorted)
    ax_inset.plot(x_sorted, intercept_sorted + slope_sorted * x_sorted, color='red', linestyle='--', label=f"Regression Line: y = {slope:.3f}x + {intercept:.3f}")
    
    # Add correspondence lines
    ax_inset.plot(x_sorted, intercept + 0.5 * x_sorted, color='green', linestyle='--', label=f"Regression Line: y = {0.5:.3f}x + {intercept:.3f}")
    ax_inset.plot(x_sorted, intercept + 0.4 * x_sorted, color='#FFA500', linestyle='--', label=f"Regression Line: y = {0.5:.3f}x + {intercept:.3f}")

    ax_inset.text(0.56, 0.03, r'$R^2$' + f': {r_value_sorted**2:.3f}', transform=ax_inset.transAxes, fontsize=12)
    ax_inset.text(0.03, 0.9, f'y = {slope_sorted:.3f}x + {intercept_sorted:.3f}', transform=ax_inset.transAxes, fontsize=12)

    ax_inset.set_xticks([])
    ax_inset.set_yticks([])

    # Save and display the plot
    save_name = os.path.join(sys.path[0], 'figures/rate_figure.pdf')
    plt.savefig(save_name, dpi=600, transparent=True, bbox_inches='tight')
    plt.show()