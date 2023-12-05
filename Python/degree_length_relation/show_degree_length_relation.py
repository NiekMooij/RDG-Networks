import numpy as np
import os
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":
    load_name = os.path.join(sys.path[0], 'data/degrees.npy')
    degrees = np.load(load_name, allow_pickle=True)
    degrees = dict(degrees)
    
    load_name = os.path.join(sys.path[0], 'data/lengths.npy')
    lengths = np.load(load_name, allow_pickle=True)
    lengths = dict(lengths)
    fig, ax1 = plt.subplots()
    
    x = np.array(list(lengths.values()))
    y = np.array(list(degrees.values()))
    
    ax1.scatter(y, 1/x, alpha=0.5)

    ax1.set_xlabel('Length', fontsize=14)
    ax1.set_ylabel('k', fontsize=14)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Save the plot (optional)
    save_name = os.path.join(sys.path[0], 'figures/degree_length_relation.pdf')
    plt.savefig(save_name, dpi=600, transparent=True, bbox_inches='tight')

    plt.show()
    