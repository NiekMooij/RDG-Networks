import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from typing import List

from Classes import LineSegment, Polygon    
from calculate_polygon_area import adjacent_polygons, adjacent_area, get_shortest_cycle
            
def get_adjacent_area_evolution(line_segments: List[LineSegment]) -> dict:
    """
    Calculate the evolution of areas of adjacent polygons over time.

    Parameters:
    - line_segments (List[LineSegment]): List of line segment objects.

    Returns:
    - dict: Dictionary containing the area evolution for each segment.
    """
    for i in range(len(line_segments)):
        line_segments[i].neighbors = {}

    line_segments_intermediate = list(line_segments[:4])
    line_segments_intermediate_dict = {segment.id: segment for segment in line_segments_intermediate}
    
    # Initialize areas dictionary
    areas = {'b1': 1, 'b2': 1, 'b3': 1, 'b4': 1}
    
    for index, segment in enumerate(line_segments):
                
        line_segments_intermediate.append(segment)
        line_segments_intermediate_dict = {segment.id: segment for segment in line_segments_intermediate}

        neighbors_initial = segment.neighbors_initial

        # Update the neighbors dictionary with neighbors_initial
        neighbors_new = {}
        neighbors_new.update(neighbors_initial)
        line_segments_intermediate_dict[segment.id].neighbors = neighbors_new
        
        # Iterate over neighbors_initial to update the neighbors of neighboring segments
        for neighbor, point in list(neighbors_initial.items()):
            # Check if the neighbor exists in line_segments_dict
            neighbors = line_segments_intermediate_dict[neighbor].neighbors
            
            neighbors_new = {}
            neighbors_new.update(neighbors)
            neighbors_new[segment.id] = point
            line_segments_intermediate_dict[neighbor].neighbors = neighbors_new
        
        line_segments_intermediate = list(line_segments_intermediate_dict.values())
        line_segments_intermediate_dict = {segment.id: segment for segment in line_segments_intermediate}

        if segment.id in ['b1', 'b2', 'b3', 'b4']:
            continue
        
        # Determine areas
        neighbor1, neighbor2 = list(segment.neighbors_initial.keys())
        
        cycle1 = get_shortest_cycle(line_segments_intermediate, segment.id, neighbor1)
        vertices1 = [line_segments_intermediate_dict[cycle1[i]].neighbors[cycle1[i + 1]] for i in range(len(cycle1) - 1)]
        
        polygon1 = Polygon(vertices=vertices1)
        polygon1.sort_vertices()
        area1 = polygon1.area()
                        
        cycle2 = get_shortest_cycle(line_segments_intermediate, segment.id, neighbor2)
        vertices2 = [line_segments_intermediate_dict[cycle2[i]].neighbors[cycle2[i + 1]] for i in range(len(cycle2) - 1)]
        
        polygon2 = Polygon(vertices=vertices2)
        polygon2.sort_vertices()
        area2 = polygon2.area()

        areas = areas.copy()
        areas[segment.id] = area1 + area2
        
        for id in cycle1[2:-2]:
            areas[id] -= area2
            
        for id in cycle2[2:-2]:
            areas[id] -= area1
                
        # Calculate and print progress percentage
        percentage = np.round((index+1) / len(line_segments[4:]) * 100, 3)
        print(f'get_adjacent_area_evolution: {percentage}% done', end='\r')

    return areas

if __name__ == "__main__":
    # Load line segments
    load_name = os.path.join(sys.path[0], 'data/segments.npy')
    line_segments = np.load(load_name, allow_pickle=True)
        
    # Get areas evolution
    areas = get_adjacent_area_evolution(line_segments)

    # Convert areas to structured array and save
    structured_array = np.array(list(areas.items()), dtype=[('key', 'U10'), ('value', object)])
    save_name = os.path.join(sys.path[0], 'data/areas.npy')
    np.save(save_name, structured_array)
