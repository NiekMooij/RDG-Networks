import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os
import sys
import networkx as nx
from typing import List, Tuple, Union, Optional

from classes import Line, LineSegment
    
def do_lines_intersect(line1: Line, line2: Line) -> Tuple[bool, Union[Tuple[float, float], None]]:
    """
    Check if two lines intersect and return the intersection point.

    Args:
    - line1 (Line): The first line segment.
    - line2 (Line): The second line segment.

    Returns:
    - intersect (bool): True if the lines intersect, False otherwise.
    - intersection_point (tuple or None): The intersection point (x, y) if lines intersect, None otherwise.
    """
    x1, y1 = line1.location
    v1, w1 = line1.direction

    x2, y2 = line2.location
    v2, w2 = line2.direction

    determinant = v1 * w2 - v2 * w1

    if determinant == 0:
        return False, (None, None)

    t1 = ((x2 - x1) * w2 - (y2 - y1) * v2) / determinant
    t2 = ((x2 - x1) * w1 - (y2 - y1) * v1) / determinant

    intersect_x = x1 + v1 * t1
    intersect_y = y2 + w2 * t2

    if -1e-6 < intersect_x < 1 + 1e-6 and -1e-6 < intersect_y < 1 + 1e-6:
        return True, (intersect_x, intersect_y)
    else:
        return False, (None, None)

def add_line_segment(line_segments: List[Line]) -> Tuple[List[Line], List[Tuple[int, int]]]:
    """
    Add a new line segment to the list of line segments and update edge information.

    Args:
    - line_segments (List[Line]): List of existing line segments.

    Returns:
    - Updated line_segments list.
    """
    location_new = (np.random.random(size=2))
    direction_new = (random.uniform(-1, 1), random.uniform(-1, 1))

    line_new = Line(location=location_new, direction=direction_new)
    intersection_points = []

    for segment in line_segments:
        location = np.array(segment.start)
        direction = np.array(segment.end) - np.array(segment.start)
        line = Line(location=location, direction=direction)

        intersect, (intersect_x, intersect_y) = do_lines_intersect(line_new, line)

        if not intersect:
            continue

        xcheck = (
            segment.end[0] <= intersect_x <= segment.start[0]
            or segment.start[0] <= intersect_x <= segment.end[0]
            or abs(intersect_x - segment.end[0]) < 1e-6
            or abs(intersect_x - segment.start[0]) < 1e-6
        )

        ycheck = (
            segment.end[1] <= intersect_y <= segment.start[1]
            or segment.start[1] <= intersect_y <= segment.end[1]
            or abs(intersect_y - segment.end[1]) < 1e-6
            or abs(intersect_y - segment.start[1]) < 1e-6
        )

        if intersect and xcheck and ycheck:
            segment_length = math.sqrt(
                (line_new.location[0] - intersect_x) ** 2
                + (line_new.location[1] - intersect_y) ** 2
            )
            intersection_points.append(
                {"id": segment.id, "point": (intersect_x, intersect_y), "segment_length": segment_length}
            )

    # Divide intersections in back and front of the new line
    intersections_b = [intersection for intersection in intersection_points if intersection["point"][0] < line_new.location[0]]
    intersections_f = [intersection for intersection in intersection_points if intersection["point"][0] > line_new.location[0]]
    
    if not intersections_b or not intersections_f:
        intersections_b = [intersection for intersection in intersection_points if intersection["point"][1] < line_new.location[1]]
        intersections_f = [intersection for intersection in intersection_points if intersection["point"][1] > line_new.location[1]]

    # Determine correct segment length
    id = str(len(line_segments) - 3)
    start = min(intersections_b, key=lambda x: x["segment_length"])
    end = min(intersections_f, key=lambda x: x["segment_length"])
    
    # Add new segment object with corresponding neighbors
    neighbors_initial = {}  
    neighbors_initial[start["id"]] = start["point"]
    neighbors_initial[end["id"]] = end["point"]
    segment_new = LineSegment(start=start["point"], end=end["point"], id=id, neighbors_initial=neighbors_initial, neighbors={})
        
    line_segments.append(segment_new)

    return line_segments

def generate_segments(size: int) -> Tuple[nx.Graph, List[LineSegment]]:
    """
    Generate a network of line segments with random intersections.

    Args:
    - size (int): Number of line segments to generate.

    Returns:
    - line_segments (List[LineSegment]): List of LineSegment objects.
    """
    borders = [
        LineSegment((1, 0), (0, 0), id='b1', neighbors_initial={'b2': (0, 0), 'b4': (1, 0)}),
        LineSegment((0, 1), (0, 0), id='b2', neighbors_initial={'b1': (0, 0), 'b3': (0, 1)}),
        LineSegment((0, 1), (1, 1), id='b3', neighbors_initial={'b2': (0, 1), 'b4': (1, 1)}),
        LineSegment((1, 1), (1, 0), id='b4', neighbors_initial={'b1': (1, 0), 'b3': (1, 1)})
    ]
    
    line_segments = borders

    for i in range(size):
        line_segments = add_line_segment(line_segments)
        
        percentage = np.round(i*(i+1) / (size*(size+1)) * 100, 3)
        print(f'generate_segments: {percentage}% done', end='\r')
                
    return line_segments

def update_segments(line_segments: List[LineSegment]) -> List[LineSegment]:
    """
    Updates a list of LineSegment objects by modifying their neighbors based on certain criteria.

    Args:
        line_segments (List[LineSegment]): A list of LineSegment objects.

    Returns:
        List[LineSegment]: A list of LineSegment objects with updated neighbor information.
    """
    line_segments_dict = {segment.id: segment for segment in line_segments}

    for id in list(line_segments_dict.keys()):
        # Get the neighbors_initial dictionary
        neighbors_initial = line_segments_dict[id].neighbors_initial

        # Update the neighbors dictionary with neighbors_initial
        neighbors_new = {}
        neighbors_new.update(neighbors_initial)
        line_segments_dict[id].neighbors = neighbors_new
        
        # Iterate over neighbors_initial to update the neighbors of neighboring segments
        for neighbor, point in list(neighbors_initial.items()):
        # Check if the neighbor exists in line_segments_dict
            neighbors = line_segments_dict[neighbor].neighbors
            
            neighbors_new = {}
            neighbors_new.update(neighbors)
            neighbors_new[id] = point
            line_segments_dict[neighbor].neighbors = neighbors_new
                
    return list(line_segments_dict.values())

def draw_segments(line_segments: List[LineSegment], fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None) -> None:
    """
    Draw the line segments on a matplotlib plot.

    Args:
    - line_segments (List[LineSegment]): List of LineSegment objects.
    - fig (Optional[plt.Figure]): Matplotlib figure to use for the plot.
    - ax (Optional[plt.Axes]): Matplotlib axes to use for the plot.
    """
    if fig is None:
        fig, ax = plt.subplots()

    for segment in line_segments:
        segment.draw(ax=ax)

    ax.hlines(0, 0, 1, color='black')
    ax.hlines(1, 0, 1, color='black')
    ax.vlines(0, 0, 1, color='black')
    ax.vlines(1, 0, 1, color='black')

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

if __name__ == "__main__":
    network_size = int(50)

    line_segments = generate_segments(network_size)
    line_segments = update_segments(line_segments)

    save_name = os.path.join(sys.path[0], 'segments.npy')
    np.save(save_name, line_segments)
    
    draw_segments(line_segments)
    
    save_name = os.path.join(sys.path[0], 'segments.pdf')
    plt.savefig(save_name, dpi=600, transparent=True, bbox_inches='tight')
    
    plt.show()