import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import networkx as nx
import matplotlib.axes._axes as axes

if __name__ == "__main__":
    # Load line segments
    load_name = os.path.join(sys.path[0], 'data/segments.npy')
    line_segments = np.load(load_name, allow_pickle=True)
        
    # Create a plot
    fig, ax1 = plt.subplots()
        
    # Calculate lengths
    lengths = {segment.id: np.linalg.norm(np.array(segment.start) - np.array(segment.end)) for segment in line_segments }

    # Convert lengths to structured array and save
    structured_array = np.array(list(lengths.items()), dtype=[('key', 'U10'), ('value', object)])
    save_name = os.path.join(sys.path[0], 'data/lengths.npy')
    np.save(save_name, structured_array)