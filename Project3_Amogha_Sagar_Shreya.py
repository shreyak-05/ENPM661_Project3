import numpy as np
import cv2
from queue import PriorityQueue

# Canvas dimensions
canvas_width = 180
canvas_height = 50

# Step size for the robot's movement
step_size = 3
clearance = 5

V = np.zeros((100, 360, 12), dtype=bool)  # 100x360 grid with 12 theta bins

# Helper function to convert (x, y, θ) into indices for `V`
def get_indices(x, y, theta):
    """Convert continuous (x, y, θ) into matrix indices."""
    x_idx = int(x / 0.5)  # Discretize X (0.5 cm per index)
    y_idx = int(y / 0.5)  # Discretize Y (0.5 cm per index)
    theta_idx = int((theta % 360) / 30)  # Convert θ into 12 bins (0° to 330°)
    return x_idx, y_idx, theta_idx

def get_indices(x, y, theta):
    """Convert continuous (x, y, θ) into matrix indices."""
    x_idx = int(x / 0.5)  # Discretize X (0.5 cm per index)
    y_idx = int(y / 0.5)  # Discretize Y (0.5 cm per index)
    theta_idx = int((theta % 360) / 30)  # Convert θ into 12 bins (0° to 330°)
    return x_idx, y_idx, theta_idx

# Define obstacle functions for "E", "N", "P", "M", "1", "6" (as per your provided functions)
def inside_E(x, y, x0=10+20, y0=35, width=10, height=20, mid_width=7, thickness=3):
    if x0 <= x <= x0 + thickness and y0 - height <= y <= y0:
        return True
    if x0 <= x <= x0 + width and y0 - height <= y <= y0 - height + thickness:
        return True
    if x0 <= x <= x0 + mid_width and y0 - height // 2 - thickness // 2 <= y <= y0 - height // 2 + thickness // 2:
        return True
    if x0 <= x <= x0 + width and y0 - thickness <= y <= y0:
        return True
    return False

def inside_N(x, y, x0=25+20, y0=35, width=10, height=20, thickness=3):
    if x0 <= x <= x0 + thickness and y0 - height <= y <= y0:
        return True
    if x0 + width - thickness <= x <= x0 + width and y0 - height <= y <= y0:
        return True
    slope = (2/3)*height / (width - 2 * thickness)
    y_s = (slope * (x - (x0 + thickness))) + (y0 - height)
    if x0 + thickness <= x <= x0 + width - thickness and y_s <= y <= y_s + height/3:
        return True
    return False

def inside_P(x, y, x0=40+20, y0=35, width=10, height=20, thickness=3):
    if x0 <= x <= x0 + thickness and y0 - height <= y <= y0:
        return True
    curve_x = x0 + thickness
    curve_y = y0 - (3 * height / 4)
    curve_r = height / 4
    if ((x - curve_x) ** 2 + ((y - curve_y) ** 2) <= curve_r ** 2 and y < y0 - height // 2 and x > x0 + thickness):
        return True
    return False

def inside_M(x, y, x0=50+20, y0=35, width=15, height=20, thickness=3):
    if x0 <= x <= x0 + thickness and y0 - height <= y <= y0:
        return True
    if x0 + width - thickness <= x <= x0 + width and y0 - height <= y <= y0:
        return True
    slope_left = (height / 2) / (width / 2 - thickness)
    y_left = slope_left * (x - x0 - thickness) + (y0 - height)
    if x0 + thickness <= x <= x0 + width / 2 and y_left <= y <= y_left + 10:
        return True
    slope_right = (-height / 2) / (width / 2 - thickness)
    y_right = slope_right * (x - (x0 + width / 2)) + (y0 - height / 2)
    if x0 + width / 2 <= x <= x0 + width - thickness and y_right <= y <= y_right + 10:
        return True
    return False

def inside_1(x, y, x0=110+20, y0=35, width=5, height=20, thickness=3):
    if x0 <= x <= x0 + thickness and y0 - height <= y <= y0:
        return True
    return False

def inside_6_second(x, y, x0=(180/2)+20, y0=35, large=20/2, med=13//2, small_radius=8/2, hole_radius=5//2, thickness=3):
    top_x = x0 + large - thickness  
    top_y = y0 - large * 1.5
    inside_top = ((x - top_x) ** 2 + (y - top_y) ** 2) <= 4 ** 2 and x <= top_x
    mid_x = x0 + large - thickness  
    mid_y = y0 - large  
    inside_middle = ((x - mid_x) ** 2 + (y - mid_y) ** 2) <= large ** 2 and x <= mid_x
    bottom_x = x0 + large - thickness  
    bottom_y = y0 - med 
    inside_bottom = ((x - bottom_x) ** 2 + (y - bottom_y) ** 2) <= med ** 2 and x >= bottom_x
    hole_x = x0 + med  
    hole_y = y0 - med 
    inside_hole = ((x - hole_x) ** 2 + (y - hole_y) ** 2) <= hole_radius ** 2  
    if (inside_bottom or inside_middle) and not inside_top and not inside_hole:
        return True
    return False

def inside_6_first(x, y, x0=(145//2)+20, y0=35, large=20/2, med=13//2, small_radius=8/2, hole_radius=5//2, thickness=3):
    top_x = x0 + large - thickness  
    top_y = y0 - large * 1.5  
    inside_top = ((x - top_x) ** 2 + (y - top_y) ** 2) <= 4 ** 2 and x <= top_x
    mid_x = x0 + large - thickness  
    mid_y = y0 - large  
    inside_middle = ((x - mid_x) ** 2 + (y - mid_y) ** 2) <= large ** 2 and x <= mid_x
    bottom_x = x0 + large - thickness  
    bottom_y = y0 - med  
    inside_bottom = ((x - bottom_x) ** 2 + (y - bottom_y) ** 2) <= med ** 2 and x >= bottom_x
    hole_x = x0 + med  
    hole_y = y0 - med  
    inside_hole = ((x - hole_x) ** 2 + (y - hole_y) ** 2) <= hole_radius ** 2  
    if (inside_bottom or inside_middle) and not inside_top and not inside_hole:
        return True
    return False

# (Include other obstacle functions here...)

def clearance_obstacles(grid_width, grid_height, clearance):
    """
    Creates an obstacle mask with a clearance region.
    - Obstacles are BLACK.
    - Clearance is BLUE.
    """
    # Initialize obstacle mask
    obstacle_mask = np.zeros((grid_height, grid_width), dtype=np.uint8)

    # Combine all letter/number obstacle checks into one list
    shapes = [inside_E, inside_N, inside_P, inside_M, inside_1, inside_6_second, inside_6_first]

    # For each shape, mark obstacles
    for y in range(grid_height):
        for x in range(grid_width):
            if any(shape(x, y) for shape in shapes):
                obstacle_mask[y, x] = 255  # Mark obstacle pixels

    # Expand clearance using OpenCV dilation
    kernel = np.ones((clearance, clearance), np.uint8)
    clearance_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)

    # Ensure obstacles remain distinct (do not overwrite obstacles)
    clearance_mask[obstacle_mask == 255] = 255

    # Mark the boundary as part of clearance
    clearance_mask[0:clearance, :] = 255  # Top boundary as clearance
    clearance_mask[-clearance:, :] = 255  # Bottom boundary as clearance
    clearance_mask[:, 0:clearance] = 255  # Left boundary as clearance
    clearance_mask[:, -clearance:] = 255  # Right boundary as clearance

    return obstacle_mask, clearance_mask


# Action functions for movement
def move_0(x, y, Θ):
    return x + step_size * np.cos(np.radians(Θ)), y + step_size * np.sin(np.radians(Θ))

def move_neg30(x, y, Θ):
    return x + step_size * np.cos(np.radians(Θ - 30)), y + step_size * np.sin(np.radians(Θ - 30))

def move_neg60(x, y, Θ):
    return x + step_size * np.cos(np.radians(Θ - 60)), y + step_size * np.sin(np.radians(Θ - 60))

def move_30(x, y, Θ):
    return x + step_size * np.cos(np.radians(Θ + 30)), y + step_size * np.sin(np.radians(Θ + 30))

def move_60(x, y, Θ):
    return x + step_size * np.cos(np.radians(Θ + 60)), y + step_size * np.sin(np.radians(Θ + 60))

# Euclidean distance function
def euclidean_distance(node1, node2):
    return np.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

# Goal checking function (within threshold)
def is_goal_reached(x, y, goal_x, goal_y):
    return euclidean_distance((x, y), (goal_x, goal_y)) <= 1.5  # Goal threshold distance of 1.5 units


# A* algorithm with live visualization of exploration
def a_star(start, goal, clearance_mask, obstacle_mask, workspace):
    open_list = PriorityQueue()
    parent = {}  # Track parents for path reconstruction

    sx, sy, stheta = start
    start_idx = get_indices(sx, sy, stheta)

    open_list.put((0, start))
    V[start_idx[1], start_idx[0], start_idx[2]] = True  # Mark start as visited

    while not open_list.empty():
        _, current = open_list.get()
        cx, cy, ctheta = current
        current_idx = get_indices(cx, cy, ctheta)

        # Check if goal is reached
        if is_goal_reached(cx, cy, goal[0], goal[1]):
            path = []
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            # Draw the final path on the workspace
            for px, py, _ in path:
                workspace[int(py), int(px)] = (0, 0, 255)  # Red color for the path
            return path[::-1]  # Reverse path order

        for action in [move_0, move_neg30, move_neg60, move_30, move_60]:
            nx, ny = action(cx, cy, ctheta)
            next_theta = (ctheta + 360) % 360  # Keep θ in range [0, 360)
            next_idx = get_indices(nx, ny, next_theta)

            # Bounds check and obstacle check
            if not (0 <= nx < canvas_width and 0 <= ny < canvas_height):  # Bounds check
                continue
            if clearance_mask[int(ny), int(nx)] == 255 or obstacle_mask[int(ny), int(nx)] == 255:
                continue
            if V[next_idx[1], next_idx[0], next_idx[2]]:  # Skip visited nodes
                continue

            # Mark as visited
            V[next_idx[1], next_idx[0], next_idx[2]] = True
            g_cost = euclidean_distance(start, (nx, ny))
            h_cost = euclidean_distance((nx, ny), goal)
            f_cost = g_cost + h_cost
            open_list.put((f_cost, (nx, ny, next_theta)))
            parent[(nx, ny, next_theta)] = current

            # Update visualization for each step (exploration) - leave a trail of explored nodes
            workspace[int(ny), int(nx)] = (0, 255, 0)  # Green for explored node
            cv2.imshow("A* Live Exploration", cv2.resize(workspace, (720, 200)))
            cv2.waitKey(1)

    return None  # No path found

# Pathfinding process (example usage with live visualization)
def visualize_path_with_arrows(workspace, path):
    for i in range(len(path) - 1):
        x1, y1, _ = path[i]
        x2, y2, _ = path[i + 1]

        # Convert to integers for OpenCV functions
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw arrows between consecutive points
        cv2.arrowedLine(workspace, (x1, y1), (x2, y2), (0, 0, 255), 1, tipLength=0.4)

    # Show the final path with arrows
    cv2.imshow("Final Path with Arrows", cv2.resize(workspace, (720, 200)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example start and goal
start = (10, 10, 0)  # Starting point (x, y, θ)
goal = (160, 40, 0)  # Goal point (x, y, θ)

# Generate obstacle and clearance masks
obstacle_mask, clearance_mask = clearance_obstacles(canvas_width, canvas_height, clearance)

# Create visualization workspace
workspace = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

# Draw obstacles in BLACK
workspace[np.where(obstacle_mask == 255)] = (0, 0, 0)  # Obstacles black

# Mark the clearance area around obstacles in blue
clearance_region = np.where((clearance_mask == 255) & (obstacle_mask == 0))

# Set clearance region to BLUE
workspace[clearance_region] = (255, 0, 0)

# Run A* search algorithm with live exploration
path = a_star(start, goal, clearance_mask, obstacle_mask, workspace)

# If a path is found, visualize it with arrows on the same map
if path:
    visualize_path_with_arrows(workspace, path)
else:
    print("No path found")

