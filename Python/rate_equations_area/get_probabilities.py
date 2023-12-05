import numpy as np
import math
import random
import os
import sys
from typing import List, Tuple, Union

from Classes import Line

def doLinesIntersect(line1: Line, line2: Line) -> Tuple[bool, Union[Tuple[float, float], None]]:
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

def get_new_hitting_ids(line_segments: List[Line]) -> Tuple[str, str]:
    """
    Add a new line segment to the list of line segments and update edge information.

    Args:
    - line_segments (List[Line]): List of existing line segments.

    Returns:
    - start_id (str): ID of the starting segment of the new line.
    - end_id (str): ID of the ending segment of the new line.
    """
    location_new = (np.random.random(size=2))
    direction_new = random.choice([(1,0), (0,1)])

    line_new = Line(location=location_new, direction=direction_new)
    intersection_points = []

    for segment in line_segments:
        location = np.array(segment.start)
        direction = np.array(segment.end) - np.array(segment.start)
        line = Line(location=location, direction=direction)

        intersect, (intersect_x, intersect_y) = doLinesIntersect(line_new, line)

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
    start = min(intersections_b, key=lambda x: x["segment_length"])
    end = min(intersections_f, key=lambda x: x["segment_length"])
    
    return start["id"], end["id"]

def get_probabilities(line_segments: List[Line], number_of_trials: int) -> dict:
    """
    Simulate line intersections and calculate hitting probabilities.

    Args:
    - line_segments (List[Line]): List of line segments.
    - number_of_trials (int): Number of simulation trials.

    Returns:
    - dict: Dictionary containing hitting probabilities for each line segment.
    """
    hitting_dict = {segment.id: 0 for segment in line_segments }
    
    for index in range(number_of_trials):
        id1, id2 = get_new_hitting_ids(line_segments)
        
        hitting_dict[id1] += 1
        hitting_dict[id2] += 1
        
        percentage = np.round((index + 1) / number_of_trials * 100, 3)
        print(f'get_probabilities: {percentage}%', end='\r')
        
    return hitting_dict

if __name__ == "__main__":
    # Load line segments
    load_name = os.path.join(sys.path[0], 'data/segments.npy')
    line_segments = np.load(load_name, allow_pickle=True)
        
    number_of_trials = 10000
    
    # Get hitting probabilities
    hitting_dict = get_probabilities(line_segments, number_of_trials)
    
    hitting_numbers = list(hitting_dict.values())
    
    # Calculate probabilities
    probabilities = {key: hitting_number / number_of_trials for key, hitting_number in hitting_dict.items() }

    # Convert probabilities to structured array and save
    structured_array = np.array(list(probabilities.items()), dtype=[('key', 'U10'), ('value', object)])
    save_name = os.path.join(sys.path[0], 'data/probabilities.npy')
    np.save(save_name, structured_array)