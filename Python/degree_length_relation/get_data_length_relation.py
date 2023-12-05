import numpy as np
import os
import sys
import networkx as nx
from typing import List

from Classes import LineSegment

def get_degrees(segments: List[LineSegment]) -> List[List[int]]:
    """
    Calculate the degree evolution of nodes in a graph over time.

    Parameters:
    - segments (List[LineSegment]): List of line segment objects.

    Returns:
    - List[List[int]]: List of lists representing the degree evolution at each time step.
    """
    # Create an empty graph.
    G = nx.Graph()

    # Add all segments minus the borders
    for index, segment in enumerate(segments):
        if segment.id in ['b1', 'b2', 'b3', 'b4']:
            continue
        
        # Add a node for the current segment
        G.add_node(segment.id)

        # Add edges between the current segment and its neighbors.
        G.add_edges_from([(segment.id, neighbor) for neighbor in segment.neighbors_initial if neighbor not in ['b1', 'b2', 'b3', 'b4']])
        
        # Calculate and print progress percentage
        percentage = np.round((index+1) / len(segments[4:]) * 100, 3)
        print(f'get_degrees: {percentage}% done', end='\r')
        
    # Calculate degrees and append to the degree evolution list
    degrees = {node: G.degree(node) for node in G.nodes()}

    return degrees

def get_lengths(segments: List[LineSegment]) -> List[List[float]]:
    """
    Calculate the lengths of line segments.

    Parameters:
    - segments (List[LineSegment]): List of line segment objects.

    Returns:
    - List[List[float]]: List of lists representing the lengths of line segments at each time step.
    """
    lengths = {segment.id: segment.length() for segment in segments if segment.id not in ['b1', 'b2', 'b3', 'b4']}
    
    return lengths

if __name__ == "__main__":
    # Load line segments
    load_name = os.path.join(sys.path[0], 'data/segments.npy')
    segments = np.load(load_name, allow_pickle=True)

    # Get degrees and lengths
    degrees = get_degrees(segments)
    lengths = get_lengths(segments)
    
    # Save degrees
    save_name_degrees = os.path.join(sys.path[0], 'data/degrees.npy')
    degrees = np.array(list(degrees.items()), dtype=[('key', 'U10'), ('value', object)])
    np.save(save_name_degrees, degrees)
    
    # Save lengths
    save_name_lengths = os.path.join(sys.path[0], 'data/lengths.npy')
    lengths = np.array(list(lengths.items()), dtype=[('key', 'U10'), ('value', object)])
    np.save(save_name_lengths, lengths)