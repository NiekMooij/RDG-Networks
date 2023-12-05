import matplotlib.pyplot as plt
import numpy as np
import math
import random
import os
import sys
import networkx as nx
from typing import List, Tuple, Union, Optional
import matplotlib.axes._axes as axes
import time
from matplotlib.patches import Polygon as polgon
import itertools
import copy

class Line:
    """
    Represents a line segment by its location and direction.

    Attributes:
    - location (Tuple[float, float]): The starting point of the line.
    - direction (Tuple[float, float]): The direction vector of the line.
    - id (Optional[Union[str, int]]): Identifier for the line segment.
    """
    
    def __init__(self, location: Tuple[float, float], direction: Tuple[float, float], id: Optional[Union[str, int]] = None):
        self.location = location
        self.direction = direction
        self.id = id

class LineSegment:
    """
    Represents a line segment defined by its start and end points.

    Attributes:
    - start (Tuple[float, float]): Starting point of the line segment.
    - end (Tuple[float, float]): Ending point of the line segment.
    - id (Optional[Union[str, int]]): Identifier for the line segment.
    """
    
    def __init__(self, start: Tuple[float, float], end: Tuple[float, float], id: Optional[Union[str, int]] = None, neighbors_initial={}, neighbors={}):
        self.start = start
        self.end = end
        self.id = id
        self.neighbors_initial = neighbors_initial
        self.neighbors = neighbors

    def draw(self, ax: axes.Axes, color: str = 'black', alpha: float = 1.0, label: bool = False):
        """
        Draw the line segment on a given axes.

        Args:
        - ax (axes.Axes): Matplotlib axes on which to draw the line segment.
        - color (str): Color of the line segment (default is 'black').
        - alpha (float): Alpha (transparency) value (default is 1.0).
        """
        
        x1, y1 = self.start
        x2, y2 = self.end
        ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=1)
        
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2, self.id, fontsize=12)
            
    def copy(self):
        """
        Create a copy of the LineSegment object.

        Returns:
        - LineSegment: A new LineSegment object with the same attributes.
        """
        return copy.deepcopy(self)
    
class Polygon:
    """
    Represents a polygon defined by a list of vertices.

    Args:
        vertices (List[Tuple[float, float]]): A list of (x, y) coordinates representing the vertices of the polygon.
    """

    def __init__(self, vertices: List[tuple]):
        """
        Initializes a Polygon instance with the provided vertices.

        Args:
            vertices (List[Tuple[float, float]]): A list of (x, y) coordinates representing the vertices of the polygon.
        """
        self.vertices = vertices

    def area(self) -> float:
        """
        Calculates the area of the polygon.

        Returns:
            float: The area of the polygon.

        Raises:
            ValueError: If the polygon has less than 3 vertices.
        """
        if len(self.vertices) < 3:
            raise ValueError("A polygon must have at least 3 vertices.")

        area = 0.0

        for i in range(len(self.vertices)):
            x1, y1 = self.vertices[i]
            x2, y2 = self.vertices[(i + 1) % len(self.vertices)]
            area += (x1 * y2) - (x2 * y1)

        area = abs(area) / 2.0

        return area

    def sort_vertices(self) -> List[Tuple[float, float]]:
        """
        Sorts the vertices of the polygon based on their polar angles with respect to a reference point.

        Returns:
            List[Tuple[float, float]]: The sorted list of vertices.
        """
        def polar_angle(point: Tuple[float, float], reference_point: Tuple[float, float]) -> float:
            """
            Calculates the polar angle of a point with respect to a reference point.

            Args:
                point (Tuple[float, float]): The coordinates (x, y) of the point for which to calculate the polar angle.
                reference_point (Tuple[float, float]): The coordinates (x, y) of the reference point.

            Returns:
                float: The polar angle in radians.
            """
            dx = point[0] - reference_point[0]
            dy = point[1] - reference_point[1]
            return np.arctan2(dy, dx)

        reference_point = min(self.vertices, key=lambda point: point[1])
        return sorted(self.vertices, key=lambda point: polar_angle(point, reference_point))

    def draw(self, ax: axes.Axes):
        """
        Draws a filled polygon with the given vertices on the specified Matplotlib axes.

        Args:
            ax (matplotlib.axes.Axes): The Matplotlib axes on which to draw the polygon.

        Note:
            This method sorts the vertices based on their polar angles with respect to a reference point
            (vertex with the lowest y-coordinate) before drawing the filled polygon.
        """
        sorted_vertices = self.sort_vertices()
        polygon = polgon(sorted_vertices, closed=True, facecolor='purple', alpha=0.5)
        ax.add_patch(polygon)
        
def remove_borders(line_segments: List[LineSegment]) -> List[LineSegment]:
    """
    Removes border line segments ('b1', 'b2', 'b3', 'b4') from a list of line segments.

    Args:
        line_segments (List[LineSegment]): A list of LineSegment objects.

    Returns:
        List[LineSegment]: A list of LineSegment objects with border segments removed.
    """
    line_segments_dict = {segment.id: segment for segment in line_segments}

    for segment_id in line_segments_dict.keys():
        neighbors = line_segments_dict[segment_id].neighbors
        # neighbors = [item for item in neighbors if item[0] not in ['b1', 'b2', 'b3', 'b4']]

        neighbors = {key: value for key, value in neighbors.items() if key not in ['b1', 'b2', 'b3', 'b4']}

        neighbors_initial = line_segments_dict[segment_id].neighbors_initial
        # neighbors_initial = [item for item in neighbors_initial if item[0] not in ['b1', 'b2', 'b3', 'b4']]
        neighbors_initial = {key: value for key, value in neighbors_initial.items() if key not in ['b1', 'b2', 'b3', 'b4']}

        line_segments_dict[segment_id].neighbors = neighbors
        line_segments_dict[segment_id].neighbors_initial = neighbors_initial

    # non_border_line_segments = [value for key, value in line_segments_dict.items() if key not in ['b1', 'b2', 'b3', 'b4']]
    non_border_line_segments = {key: value for key, value in line_segments_dict.items() if key not in ['b1', 'b2', 'b3', 'b4']}

    return non_border_line_segments

def generate_network(line_segments: List[LineSegment]) -> nx.Graph:
    """
    Generates a network (graph) based on a list of LineSegment objects.

    Args:
        line_segments (List[LineSegment]): A list of LineSegment objects.

    Returns:
        nx.Graph: A NetworkX graph representing the network of LineSegment objects.
    """
    # Create an empty graph.
    G = nx.Graph()
    
    # Add all segments minus the borders
    for segment in line_segments:

        G.add_node(segment.id)
        
        # Add edges between the current segment and its neighbors.
        G.add_edges_from([(segment.id, key) for key, neighbor in segment.neighbors_initial.items()])
    
    return G

def draw_segments(line_segments: List[LineSegment], fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None, label=False):
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
        segment.draw(ax=ax, label=True)
        # ax.text((segment.start[0] + segment.end[0]) / 2, (segment.start[1] + segment.end[1]) / 2, segment.id, fontsize=12)

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

def is_clockwise_outside(segment1: LineSegment, segment2: LineSegment):
    # Calculate vectors for each segment
    vector1 = (segment1.end[0] - segment1.start[0], segment1.end[1] - segment1.start[1])
    vector2 = (segment2.end[0] - segment2.start[0], segment2.end[1] - segment2.start[1])

    # Calculate the cross product of the two vectors
    cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]

    # Check the sign of the cross product
    if cross_product < 0:
        return 'Clockwise'
    elif cross_product > 0:
        return 'Counterclockwise'
    else:
        return 'Collinear'

def fraction_point_on_segment(linesegment: LineSegment, point: Tuple[float, float]) -> float:
    """
    Calculate the fraction of the line segment that a given point is on.

    Args:
    - point (Tuple[float, float]): The point to check.

    Returns:
    - float: The fraction of the line segment that the point is on, as a value between 0 and 1.
    """
    x1, y1 = linesegment.start
    x2, y2 = linesegment.end
    x, y = point

    # Calculate the lengths of the line segment and the two sub-segments
    segment_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    subsegment1_length = ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
    subsegment2_length = ((x2 - x) ** 2 + (y2 - y) ** 2) ** 0.5

    # Calculate the fraction
    fraction = subsegment1_length / segment_length

    # Ensure the fraction is within the range [0, 1]
    fraction = max(0, min(1, fraction))

    return fraction

def distance_to_point(point, item):
    _, value = item
    x, y = value
    return math.sqrt((x - point[0])**2 + (y - point[1])**2)
    
def is_point_on_line(point, Line: Line):
    
    if Line.direction[0] != 0:
        m = Line.direction[1] / Line.direction[0]    
        x = point[0]
        b = Line.location[1] - m * Line.location[0]
        expected_y = m * x + b
        return abs(point[1] - expected_y) < 1e-6
        
    else:
        if point[0] == Line.location[0]:
            return True

def get_shortest_cycle(line_segments: List[LineSegment], segment_first_id: str, segment_second_id: str, orientation: str='Clockwise'):
    
    # If we can not go clockwise we flip the borders. Can only occur with borders anyway.
    if (segment_first_id, segment_second_id) in [('b2', 'b1'), ('b3', 'b2'), ('b4', 'b3'), ('b1', 'b4')]:
        segment_first_id, segment_second_id = segment_second_id, segment_first_id
    
    # Determine the first linesegments and first node
    line_segments_dict = { segment.id: segment for segment in line_segments }    
    segment_first = line_segments_dict[segment_first_id]
    
    # Second segment must be connected to the first segment
    if segment_second_id not in list(segment_first.neighbors.keys()):
        print('\nSecond line is not adjacent to first line!\n')
        exit()

    else:
        point = segment_first.neighbors[segment_second_id]        
                            
    # In case we start at the endpoint we need to set orientation correctly  
    if segment_second_id in list(line_segments_dict[segment_first_id].neighbors_initial.keys()):

        if segment_first.end != segment_first.neighbors[segment_second_id]:
            segment_first.start, segment_first.end = segment_first.end, segment_first.start
        
        segment_second = line_segments_dict[segment_second_id].copy()

        # Orientate the second line correctly wrt the first line
        if is_clockwise_outside(segment_first, segment_second) != orientation:
            segment_second.start, segment_second.end = segment_second.end, segment_second.start

    # In case we are not at an endpoint we still need to check if the line segment is oriented correctly
    else:
        segment_second = line_segments_dict[segment_second_id].copy()

        # Check if the line segment is oriented correctly and change if necessary
        line = Line(location=segment_first.start, direction=np.array(segment_first.end) - np.array(segment_first.start))                         
        if is_point_on_line(segment_second.end, line):
            segment_second.start, segment_second.end = segment_second.end, segment_second.start

    # Add correct oriented first line to the list
    segments_visited = [segment_first, segment_second]     
    segment_initial = segment_second.copy()
    
    
    while True:

        neighbors = segment_initial.neighbors
        
        sorted_data = dict(sorted(neighbors.items(), key=lambda item: distance_to_point(point, item)))
        neighbors_all = list(sorted_data.keys())
        
        # We only have to check line segments that are ahead on the current line segment. Note that this depends on the orientation of the line segment
        arr = []
        fraction = fraction_point_on_segment(segment_initial, segment_initial.neighbors[segments_visited[-2].id])

        for n in neighbors_all:

            point = segment_initial.neighbors[n]
            fraction_n = fraction_point_on_segment(segment_initial, point)  

            if fraction_n > fraction:
                arr.append(n)

        neighbors_all = arr
        
        # ----------------------------------------------------------------------------------------------------------------

        # For every neighbor we check if the line segment moves in the correct orientation
        neighbors_clock = []
        for n in neighbors_all:

            # Exception if neighbor is at the end of the line
            if n in list(segment_initial.neighbors_initial.keys()):

                segment_start, segment_end = line_segments_dict[n].start, line_segments_dict[n].end
                point = segment_initial.neighbors[n]

                n1 = line_segments_dict[n].copy()
                n1.start, n1.end = point, segment_end
                
                n2 = line_segments_dict[n].copy()
                n2.start, n2.end = point, segment_start

                # Note that only one of the options can be true
                if is_clockwise_outside(segment_initial, n1) == orientation:
                    n1.start = segment_start
                    neighbors_clock.append(n1)

                if is_clockwise_outside(segment_initial, n2) == orientation:
                    n2.start = segment_end
                    neighbors_clock.append(n2)
                    
            else:     
                segment_new = line_segments_dict[n].copy()       
                segment_old = segments_visited[-1].copy()

                # Orientate the segment in the corrext direction
                if segment_new.start != segment_old.neighbors[segment_new.id]:
                    segment_new.start, segment_new.end = segment_old.neighbors[segment_new.id], segment_new.start

                if is_clockwise_outside(segment_old, segment_new) == orientation:
                    neighbors_clock.append(segment_new)
                            
        segment_next = neighbors_clock[0].copy()
        point = segment_next.neighbors[segments_visited[-1].id]
        
        if segment_next.id == segment_first_id:
            break

        if is_clockwise_outside(segment_next, segments_visited[-1]) == orientation: # Change back to get working code
            segment_next.start, segment_next.end = segment_next.end, segment_next.start

        segments_visited.append(segment_next)
        segment_initial = segment_next.copy()        

    segments_visited.append(segments_visited[0])

    return [ segment.id for segment in segments_visited ]
    
def adjacent_polygons(line_segments: List[LineSegment], segment_id: str) -> List[Polygon]:
    """
    Calculate adjacent polygons based on the given segment ID.

    Args:
    segment_id (str): The ID of the segment for which adjacent polygons are calculated.

    Returns:
    List[Polygon]: A list of Polygon objects representing adjacent polygons.
    """
    
    line_segments_dict = {segment.id: segment for segment in line_segments}
    neighbors = list(line_segments_dict[segment_id].neighbors.keys())

    if segment_id == 'b1':
        neighbors = [item for item in neighbors if item != 'b4']
    if segment_id == 'b2':
        neighbors = [item for item in neighbors if item != 'b1']
    if segment_id == 'b3':
        neighbors = [item for item in neighbors if item != 'b2']
    if segment_id == 'b4':
        neighbors = [item for item in neighbors if item != 'b3']

    polygon_arr = []
    for n in neighbors:
        cycle = get_shortest_cycle(line_segments, segment_id, n)

        vertices = []
        for i in range(len(cycle) - 1):
            vertex = line_segments_dict[cycle[i]].neighbors[cycle[i + 1]]
            vertices.append(vertex)

        polygon = Polygon(vertices)
        polygon.sort_vertices()
        polygon_arr.append(polygon)

    return polygon_arr
    
def draw_adjacent_polygons(polygon_arr: List[Polygon], ax: plt.Axes) -> None:
    """
    Draw the adjacent polygon formed by two line segments.

    Args:
    segment_first_id (str): The ID of the first line segment.
    segment_second_id (str): The ID of the second line segment.
    ax (matplotlib.axes.Axes): The matplotlib axes on which to draw the polygon.

    Returns:
    None
    """

    for polygon in polygon_arr:
        polygon.draw(ax)
        
def adjacent_area(polygon_arr: List[Polygon]) -> float:
    """
    Calculate the total area of adjacent polygons formed by the given segment.

    Args:
    segment_first_id (str): The ID of the segment for which adjacent area is calculated.

    Returns:
    float: The total area of adjacent polygons.
    """
    area = 0.0
    for polygon in polygon_arr:
        area += polygon.area()

    return area

def draw_network(G: nx.Graph, ax: plt.Axes) -> None:
    """
    Draw a network graph on the given matplotlib axes.

    Args:
    G (nx.Graph): The NetworkX graph to be drawn.
    ax (matplotlib.axes.Axes): The matplotlib axes on which to draw the network.

    Returns:
    None
    """
    pos = nx.spring_layout(G)
    colors = [G.nodes[node]['color'] if 'color' in G.nodes[node] else 'cornflowerblue' for node in G.nodes]
    nx.draw(G, pos, node_color=colors, with_labels=True, ax=ax)

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    
if __name__ == '__main__':
    data_name = 'segments.npy'
    
    # data_name = 'segments_check.npy'
    # data_name = 'segments_check_check.npy'
    
    load_name = os.path.join(sys.path[0], data_name)
    line_segments = np.load(load_name, allow_pickle=True)
    
    check = line_segments[10]
    print(check.neighbors)
    
    # exit()
    # Initialize figure
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8,8))

    # Set the segment ID for which to calculate adjacent polygons
    segment_id = '1'
    segment_second_id = '4'
    
    start_time = time.time()
    
    cycle = get_shortest_cycle(line_segments, segment_first_id=segment_id, segment_second_id=segment_second_id)
    print(f'cycle: {cycle}')
    
    line_segments_dict = { segment.id: segment for segment in line_segments }
    
    vertices = []
    for i in range(len(cycle) - 1):
        vertex = line_segments_dict[cycle[i]].neighbors[cycle[i + 1]]
        vertices.append(vertex)

    polygon = Polygon(vertices)
    polygon.sort_vertices()
    
    polygon.draw(ax1)
    
    # ------------------------------------------------------

    # Draw segments
    draw_segments(line_segments, fig=fig, ax=ax1, label=True)
    # line_segments_dict = { segment.id: segment for segment in line_segments }    
    # line_segments_dict[segment_id].draw(ax1, color='red', label=True)
    
    # # Draw adjacent polygons
    # polygon_arr = adjacent_polygons(line_segments, segment_id)
    # draw_adjacent_polygons(polygon_arr, ax1)
    # area = adjacent_area(polygon_arr)

    # # Print the adjacent area
    # print(f'\nArea: {area}')

    plt.show()