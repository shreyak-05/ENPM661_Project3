# -*- coding: utf-8 -*-
import numpy as np  # Import numpy for numerical operations and array manipulation
import cv2  # Import OpenCV for image processing tasks like drawing and dilation
from queue import PriorityQueue  # Import PriorityQueue for implementing the A* search algorithm
import time  # Import time for measuring execution time of the algorithm

# Canvas dimensions
canvas_width = 600  # Define the width of the canvas
canvas_height = 250  # Define the height of the canvas

# Step size for the robot's movement
step_size = int(input("Input step size between 1 and 10: "))  # Get the step size from the user
clearance = int(input("Enter clearance: "))  # Get the clearance value from the user
robot_radius = 5  # Define the robot's radius (in mm)

V = np.zeros((500, 1200, 12), dtype=bool)  # Initialize a 3D boolean array to track visited nodes (x, y, theta)

def check_accessibility(start, goal, obstacle_mask, clearance_mask, canvas_width, canvas_height):
    """Checks if the start and goal nodes are accessible (not in obstacles or clearance zones)."""
    def is_valid_node(node, mask):
        """Helper function to check if a node is within bounds and not in an obstacle."""
        x, y = int(node[0]), int(node[1])  # Extract x and y coordinates from the node
        return 0 <= x < canvas_width and 0 <= y < canvas_height and mask[y, x] == 0  # Check if within bounds and not in mask

    return is_valid_node(start, obstacle_mask) and is_valid_node(goal, obstacle_mask) and \
           is_valid_node(start, clearance_mask) and is_valid_node(goal, clearance_mask)  # Check both start and goal

def get_indices(x, y, theta):
    """Convert continuous (x, y, θ) into discrete matrix indices."""
    return int(x / 0.5), int(y / 0.5), int((theta % 360) / 30)  # Convert to grid indices

def E(x, y, x0=100, y0=175, width=30, height=100, mid_width=21, thickness=9):
    """Defines the 'E' shaped obstacle region."""
    return (x0 <= x <= x0 + thickness and y0 - height <= y <= y0) or \
           (x0 <= x <= x0 + width and y0 - height <= y <= y0 - height + thickness) or \
           (x0 <= x <= x0 + mid_width and y0 - height // 2 - thickness // 2 <= y <= y0 - height // 2 + thickness // 2) or \
           (x0 <= x <= x0 + width and y0 - thickness <= y <= y0)

def N(x, y, x0=150, y0=175, width=30):
    """Defines the 'N' shaped obstacle region."""
    if x0 <= x <= x0 + 9 and y0 - 100 <= y <= y0:
        return True
    if x0 + width - 9 <= x <= x0 + width and y0 - 100 <= y <= y0:
        return True
    slope = (2/3)*100 / (width - 2 * 9)
    y_s = (slope * (x - (x0 + 9))) + (y0 - 100)
    return x0 + 9 <= x <= x0 + width - 9 and y_s <= y <= y_s + 100/3

def P(x, y, x0=200, y0=175, width=30):
    """Defines the 'P' shaped obstacle region."""
    if x0 <= x <= x0 + 9 and y0 - 100 <= y <= y0:
        return True
    curve_x, curve_y, curve_r = x0 + 9, y0 - (3 * 100 / 4), 100 / 4
    return ((x - curve_x) ** 2 + ((y - curve_y) ** 2) <= curve_r ** 2 and y < y0 - 100 // 2 and x > x0 + 9)

def M(x, y, x0=250, y0=175):
    """Defines the 'M' shaped obstacle region."""
    if x0 <= x <= x0 + 9 and y0 - 100 <= y <= y0:
        return True
    if x0 + 45 - 9 <= x <= x0 + 45 and y0 - 100 <= y <= y0:
        return True
    slope_left = (100 / 2) / (45 / 2 - 9)
    y_left = slope_left * (x - x0 - 9) + (y0 - 100)
    slope_right = (-100 / 2) / (45 / 2 - 9)
    y_right = slope_right * (x - (x0 + 45 / 2)) + (y0 - 100 / 2)
    return (x0 + 9 <= x <= x0 + 45 / 2 and y_left <= y <= y_left + 10) or \
           (x0 + 45 / 2 <= x <= x0 + 45 - 9 and y_right <= y <= y_right + 10)

def sixf(x, y, x0=330, y0=175):
    """Defines the '6f' shaped obstacle region."""
    carve_x_out, carve_y_out, carve_out_r = x0 + 41, y0 - 75, 17
    main_x, main_y, main_r = x0 + 41, y0 - 50, 50
    minor_x, minor_y, minor_r = x0 + 41, y0 - 30, 30
    h_x, h_y, hole_r = x0 + 40, y0 - 30, 15
    carve_out = ((x - carve_x_out) ** 2 + (y - carve_y_out) ** 2) <= carve_out_r ** 2 and x <= carve_x_out
    left_semi = ((x - main_x) ** 2 + (y - main_y) ** 2) <= main_r ** 2 and x <= main_x
    right_semi = ((x - minor_x) ** 2 + (y - minor_y) ** 2) <= minor_r ** 2 and x >= minor_x
    hole = ((x - h_x) ** 2 + (y - h_y) ** 2) <= hole_r ** 2
    return (left_semi or right_semi) and not carve_out and not hole

def sixs(x, y, x0=420, y0=175):
    """Defines the '6s' shaped obstacle region."""
    carve_x_out, carve_y_out, carve_out_r = x0 + 41, y0 - 75, 17
    main_x, main_y, main_r = x0 + 41, y0 - 50, 50
    minor_x, minor_y, minor_r = x0 + 41, y0 - 30, 30
    h_x, h_y, hole_r = x0 + 40, y0 - 30, 15
    carve_out = ((x - carve_x_out) ** 2 + (y - carve_y_out) ** 2) <= carve_out_r ** 2 and x <= carve_x_out
    left_semi = ((x - main_x) ** 2 + (y - main_y) ** 2) <= main_r ** 2 and x <= main_x
    right_semi = ((x - minor_x) ** 2 + (y - minor_y) ** 2) <= minor_r ** 2 and x >= minor_x
    hole = ((x - h_x) ** 2 + (y - h_y) ** 2) <= hole_r ** 2
    return (left_semi or right_semi) and not carve_out and not hole

def one(x, y, x0=520, y0=175, thickness=9):
    """Defines the '1' shaped obstacle region."""
    return x0 <= x <= x0 + thickness and y0 - 100 <= y <= y0

def clearance_obstacles(grid_width, grid_height, clearance):
    """Creates obstacle and clearance masks using OpenCV dilation."""
    obstacle_mask = np.zeros((grid_height, grid_width), dtype=np.uint8)  # Initialize obstacle mask
    shapes = [E, N, P, M, sixs, sixf, one]  # List of obstacle shape functions

    for y in range(grid_height):
        for x in range(grid_width):
            if any(shape(x, y) for shape in shapes):
                obstacle_mask[y, x] = 255  # Mark obstacle pixels

    kernel = np.ones((clearance, clearance), np.uint8)  # Create a kernel for dilation
    clearance_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)  # Dilate the obstacle mask
    clearance_mask[obstacle_mask == 255] = 255  # Ensure obstacles are included in clearance

    # Add clearance to the boundaries
    clearance_mask[0:clearance, :] = 255
    clearance_mask[-clearance:, :] = 255
    clearance_mask[:, 0:clearance] = 255
    clearance_mask[:, -clearance:] = 255

    return obstacle_mask, clearance_mask

def move(x, y, θ, step_size, clearance_mask, obstacle_mask, angle_change):
    """Calculates the new position after a move with a given angle change."""
    new_theta = (θ + angle_change) % 360  # Calculate the new theta
    nx = x + step_size * np.cos(np.radians(new_theta))  # Calculate new x
    ny = y + step_size * np.sin(np.radians(new_theta))  # Calculate new y

    if 0 <= nx < canvas_width and 0 <= ny < canvas_height:
        if clearance_mask[int(ny), int(nx)] == 255 or obstacle_mask[int(ny), int(nx)] == 255:
            return None  # Return None if new position is in obstacle or clearance
        return nx, ny, new_theta  # Return new position
    return None  # Return None if new position is out of bounds

def check_line_clearance(x1, y1, x2, y2, clearance_mask, obstacle_mask):
    """Checks if the line segment between two points is clear of obstacles and clearance."""
    num_samples = 5  # Number of samples along the line
    for i in range(1, num_samples + 1):
        t = i / num_samples
        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))
        if clearance_mask[y, x] == 255 or obstacle_mask[y, x] == 255:
            return False  # Return False if any sample is in obstacle or clearance
    return True  # Return True if line is clear

def euclidean_distance(n1, n2):
    """Calculates the Euclidean distance between two points."""
    return np.sqrt((n1[0] - n2[0])**2 + (n1[1] - n2[1])**2)

def is_goal_reached(x, y, goal_x, goal_y):
    """Checks if the robot has reached the goal within a tolerance."""
    return euclidean_distance((x, y), (goal_x, goal_y)) <= 1.5

def reconstruct_path(came_from, current, workspace):
    """Reconstructs the path from the goal to the start using the came_from dictionary."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    for px, py, _ in path:
        workspace[int(py), int(px)] = (0, 0, 255)  # Draw the path on the workspace
    return path

def is_valid_position(x, y, theta, clearance_mask, obstacle_mask, V, canvas_width, canvas_height, cx, cy):
    """Checks if a position is valid for exploration."""
    next_idx = get_indices(x, y, theta)
    return 0 <= x < canvas_width and 0 <= y < canvas_height and \
           clearance_mask[int(y), int(x)] == 0 and obstacle_mask[int(y), int(x)] == 0 and \
           check_line_clearance(cx, cy, x, y, clearance_mask, obstacle_mask) and \
           not V[next_idx[1], next_idx[0], next_idx[2]]

def get_neighbors(current, step_size, clearance_mask, obstacle_mask, V, canvas_width, canvas_height):
    """Generates valid neighbor nodes for a given node."""
    cx, cy, ctheta = current
    neighbors = []
    for angle_change in [0, -30, -60, 30, 60]:  # Try different angle changes
        next_pos = move(cx, cy, ctheta, step_size, clearance_mask, obstacle_mask, angle_change)
        if next_pos and is_valid_position(*next_pos, clearance_mask, obstacle_mask, V, canvas_width, canvas_height, cx, cy):
            neighbors.append(next_pos)
    return neighbors

def a_star(start, goal, clearance_mask, obstacle_mask, workspace):
    """Implements the A* search algorithm to find a path from start to goal."""
    open_list = PriorityQueue()  # Initialize the open list as a priority queue
    parent = {}  # Dictionary to store parent nodes for path reconstruction
    g_costs = {}  # Dictionary to store g-costs (cost from start to current node)
    sx, sy, stheta = start
    start_idx = get_indices(sx, sy, stheta)

    open_set = set()  # Set to track nodes in the open list
    open_list.put((0, start))  # Add the start node to the open list
    open_set.add(start)
    V[start_idx[1], start_idx[0], start_idx[2]] = True  # Mark start node as visited
    g_costs[start] = 0  # Initialize g-cost of start node to 0

    while not open_list.empty():
        _, current = open_list.get()  # Get the node with the lowest f-cost
        open_set.remove(current)  # Remove the current node from the open set
        cx, cy, ctheta = current
        current_idx = get_indices(cx, cy, ctheta)

        if is_goal_reached(cx, cy, goal[0], goal[1]):
            return reconstruct_path(parent, current, workspace)  # Return the reconstructed path

        for neighbor in get_neighbors(current, step_size, clearance_mask, obstacle_mask, V, canvas_width, canvas_height):
            nx, ny, next_theta = neighbor
            next_idx = get_indices(nx, ny, next_theta)
            g_cost = g_costs[current] + step_size  # Calculate g-cost of the neighbor
            h_cost = euclidean_distance((nx, ny), goal)  # Calculate h-cost (heuristic)
            f_cost = g_cost + h_cost  # Calculate f-cost

            if neighbor not in g_costs or g_cost < g_costs[neighbor]:
                parent[neighbor] = current  # Update parent of the neighbor
                g_costs[neighbor] = g_cost  # Update g-cost of the neighbor
                if neighbor not in open_set:
                    open_list.put((f_cost, neighbor))  # Add the neighbor to the open list
                    open_set.add(neighbor)
                V[next_idx[1], next_idx[0], next_idx[2]] = True  # Mark neighbor as visited
                workspace[int(ny), int(nx)] = (0, 255, 0)  # Draw the explored node
                cv2.imshow("A* Live Exploration", cv2.resize(workspace, (600, 250)))  # Show the exploration
                cv2.waitKey(1)  # Small delay for visualization
                if is_goal_reached(nx, ny, goal[0], goal[1]):
                    return reconstruct_path(parent, neighbor, workspace)  # Return path if goal is reached

    return None  # Return None if no path is found

def visualize_path_with_lines(workspace, path):
    """Visualizes the final path with lines on the workspace."""
    for i in range(len(path) - 1):
        x1, y1, _ = path[i]
        x2, y2, _ = path[i + 1]
        cv2.line(workspace, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Draw the line segment
    cv2.imshow("Final Path with Lines", cv2.resize(workspace, (600, 250)))  # Show the final path
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_start_goal_inputs(obstacle_mask, clearance_mask):
    """Gets start and goal node inputs from the user with validation."""
    def get_input(node_type):
        while True:
            try:
                print(f"\n===== {node_type.upper()} NODE =====")
                x = float(input(f"Enter {node_type} X (0-600): "))
                y = float(input(f"Enter {node_type} Y (0-250): "))
                y = canvas_height - y  
                theta = float(input(f"Enter {node_type} Theta (multiple of 30, 0-360): "))
                if not (0 <= x <= 600 and 0 <= y <= 250):
                    print("Out of range.")
                    continue
                if x < 5 or x > 595 or y > 245 or y < 5:
                    print("In boundary clearance.")
                    continue
                if obstacle_mask[int(y), int(x)] == 255:
                    print("In obstacle.")
                    continue
                if clearance_mask[int(y), int(x)] == 255 and obstacle_mask[int(y), int(x)] == 0:
                    print("In obstacle clearance zone.")
                    continue
                if theta % 30 != 0 or not (0 <= theta <= 360):
                    print("Theta must be a multiple of 30 and within 0-360 degrees.")
                    continue
                theta = theta % 360
                print(f"{node_type.capitalize()} position ({x:.1f}, {y:.1f}, {theta:.1f}°) is valid.")
                return (x, y, theta)
            except ValueError:
                print("Invalid input.")
    print("\n===== PATH PLANNING CONFIGURATION =====")
    print("Canvas dimensions: 600x250")
    print("Please enter coordinates between (5-595, 5-245) to avoid boundary clearance.\n")
    start = get_input("start")
    goal = get_input("goal") if start else None
    return start, goal

obstacle_mask, clearance_mask = clearance_obstacles(canvas_width, canvas_height, clearance)  # Generate obstacle and clearance masks
workspace = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # Initialize workspace with white
workspace[np.where(obstacle_mask == 255)] = (0, 0, 0)  # Draw obstacles in black
workspace[np.where((clearance_mask == 255) & (obstacle_mask == 0))] = (255, 0, 0)  # Draw clearance zones in blue

start, goal = get_start_goal_inputs(obstacle_mask, clearance_mask)  # Get start and goal nodes from the user

cv2.circle(workspace, (int(start[0]), int(start[1])), 4, (255, 255, 0), -1)  # Draw start node
cv2.circle(workspace, (int(goal[0]), int(goal[1])), 4, (255, 255, 0), -1)  # Draw goal node

if check_accessibility(start, goal, obstacle_mask, clearance_mask, canvas_width, canvas_height):
    # print("Yes for both")
    start_time = time.time()
    path = a_star(start, goal, clearance_mask, obstacle_mask, workspace)  # Run A* algorithm
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    if path:
        visualize_path_with_lines(workspace, path)  # Visualize the final path
    else:
        print("No path found")
else:
    print("Start and goal are not accessible")
