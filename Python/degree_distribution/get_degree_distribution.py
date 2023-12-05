import numpy as np
import os
import sys
import networkx as nx
from typing import List

from Classes import LineSegment

def get_degrees(line_segments: List[LineSegment]) -> List[List[int]]:
    """
    Calculate the degree evolution of nodes in a graph over time.

    Parameters:
    - line_segments (List[LineSegment]): List of line segment objects.

    Returns:
    - List[List[int]]: List of lists representing the degree evolution at each time step.
    """
    # Create an empty graph.
    G = nx.Graph()

    # Add all segments minus the borders
    for index, segment in enumerate(line_segments):
        if segment.id in ['b1', 'b2', 'b3', 'b4']:
            continue
        
        # Add a node for the current segment
        G.add_node(segment.id)

        # Add edges between the current segment and its neighbors.
        G.add_edges_from([(segment.id, neighbor) for neighbor in segment.neighbors_initial if neighbor not in ['b1', 'b2', 'b3', 'b4']])
        
        # Calculate and print progress percentage
        percentage = np.round((index+1) / len(line_segments[4:]) * 100, 3)
        print(f'get_degree_evolution: {percentage}% done', end='\r')
        
    # Calculate degrees and append to the degree evolution list
    degrees = [degree for _, degree in G.degree]

    return degrees

if __name__ == "__main__":
    load_name = os.path.join(sys.path[0], 'data/segments.npy')
    line_segments = np.load(load_name, allow_pickle=True)

    degrees = get_degrees(line_segments)
    
    save_name = os.path.join(sys.path[0], 'data/degrees.npy')
    np.save(save_name, degrees)
