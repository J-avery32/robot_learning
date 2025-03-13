import time
import numpy as np
from create_grid_from_png import grid_from_image
import heapq

# Got these functions from here:
# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    denom = np.linalg.norm(vector)
    return vector / max(denom, 1e-16)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.round(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)), 5)




def print_coords(matrix: np.ndarray):
    print(np.flip(matrix, axis=1).T)


np.set_printoptions(linewidth=130)

# Map and simulation parameters
WIDTH = 25
HEIGHT = WIDTH
NUM_STATES = WIDTH*HEIGHT
TARGET_LOCATION = [11,5]
BEGINNING_LOCATION = [23,20]
WIND_DIRECTION = [1, 1]

# 1s in this grid are land and 0s in this grid are water
LAKE_GRID = grid_from_image("lake_obstacle.png")

POLAR_DIAGRAM_VELOCITIES = {
 angle_between((1,0),(0,1)): 3.5,
 angle_between((1,0),(1,0)): 2.25,
 angle_between((-1,1),(1,0)): 3,
 angle_between((1,1),(1,0)): 3,
}

# to get the velocity from one state to the other do this: 
# S_final - S_init = action, and then do: POLAR_DIAGRAM_VELOCITIES[angle_between(action, WIND_DIRECTION)]
# Remember to use 1/v as your cost

# Remember that nodes are not connected if S_final - S_init = WIND_DIRECTION

# Land nodes are not connected to anything

# When you have your path mark each node in the path with something like 8 in the LAKE_GRID array and then
# print it out for debugging

def is_valid_move(x, y, current, grid):
    """ Returns whether a move to (x, y) is valid and doesn't go against the wind direction. """
    if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
        return False  # Out of bounds
    
    if grid[x, y] == 1:
        return False  # Hits the ground

    # Calculate direction vector from current position to (x, y)
    direction = np.array([x - current[0], y - current[1]])

    # Goes against the wind
    if direction[0] == -WIND_DIRECTION[0] and direction[1] == -WIND_DIRECTION[1]:
        return False
    return True

def heuristic(a, b):
    return 0

def astar(start, goal, grid):
    """ Implements A* algorithm to find the shortest path. """
    open_list = []
    closed_list = set()
    came_from = {}
    # Directions for 8-connected grid
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # g(n) = cost from start to current node
    g_score = {start: 0}
    
    # f(n) = g(n) + h(n)
    f_score = {start: heuristic(start, goal)}

    # Push the start position into the priority queue (min-heap)
    heapq.heappush(open_list, (f_score[start], start))

    while open_list:
        _, current = heapq.heappop(open_list)
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        closed_list.add(current)

        # Explore neighbors
        for direction in DIRECTIONS:
            neighbor = (current[0] + direction[0], current[1] + direction[1])

            if not is_valid_move(neighbor[0], neighbor[1], current, grid) or neighbor in closed_list:
                continue

            # Calculate movement cost
            action = np.array(neighbor) - np.array(current)
            angle = angle_between(action, WIND_DIRECTION)
            if angle in POLAR_DIAGRAM_VELOCITIES:
                velocity = POLAR_DIAGRAM_VELOCITIES[angle]
                factor = 1
                if abs(action[0]) * abs(action[1]) == 1:
                    factor = 1.41
                cost = factor / velocity  # Cost is the inverse of velocity
            else:
                assert False

            tentative_g_score = g_score[current] + cost

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                came_from[neighbor] = current
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None  # If no path found

def mark_path(path, grid):
    """ Mark the path on the grid with value 8. """
    cost = 0
    for i in range(len(path)-1):
        action = np.array(path[i+1]) - np.array(path[i])
        angle = angle_between(action, WIND_DIRECTION)
        if angle in POLAR_DIAGRAM_VELOCITIES:
            velocity = POLAR_DIAGRAM_VELOCITIES[angle]
            factor = 1
            if abs(action[0]) * abs(action[1]) == 1:
                factor = 1.41
            cost += factor / velocity  # Cost is the inverse of velocity
        grid[path[i][0], path[i][1]] = 8
    print("Cost:", cost)
    return grid
t0 = time.time()
# Run the A* algorithm
start = tuple(BEGINNING_LOCATION)
goal = tuple(TARGET_LOCATION)

path = astar(start, goal, LAKE_GRID)

if path:
    print("Path found:")
    print(path)
    LAKE_GRID = mark_path(path, LAKE_GRID)
    LAKE_GRID[start] = 8
    LAKE_GRID[goal] = 8
    print_coords(LAKE_GRID)
else:
    print("No path found.")

print("TIME: ", time.time() - t0)