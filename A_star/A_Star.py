# Import necessary libraries
from queue import Queue
import numpy as np
from sortedcollections import OrderedSet
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import time
import pygame
import math

# Function to generate circle coordinates for the letter "P"
def generate_p_coordinates(radius, center_x, center_y):
    """
    Generates circle coordinates for a letter "P" with given radius and width.
    - radius: Radius of the semicircle forming the top part of "P"
    - center_x, center_y: Positioning the letter "P" in the coordinate space
    """ 
    semicircle_offsetcoordinates = []
    semicircle_coordinates = []
    for theta in np.linspace(-np.pi/2, np.pi/2, 20):  # Generate half-circle
        x = center_x + radius * np.cos(theta) 
        x_offset = x+20
        y = center_y + radius * np.sin(theta)
        if theta>0:
            y_offset = y+20
        else:
            y_offset = y-20
        semicircle_offsetcoordinates.append([x_offset, y_offset])
        semicircle_coordinates.append([x, y])

    return semicircle_coordinates, semicircle_offsetcoordinates

# Function to generate upper circle coordinates for a digit "6"
def generate_6_circle(radius, center_x, center_y):
    """
    Generates upper circle coordinates for a digit "6" with given radius.
    - radius: Radius of the semicircle forming the top part of "6"
    - center_x, center_y: Positioning the digit "6" in the coordinate space
    """ 
    semicircle_coordinates = []
    semicircle_offsetcoordinates = []

    for theta in np.linspace(-np.pi, np.pi, 20): 
        x = center_x + radius * np.sin(theta)   
        x_offset = center_x + (radius + 20) * np.sin(theta)
        y = center_y - radius * np.cos(theta)
        y_offset = center_y - (radius + 20) * np.cos(theta)
        
        semicircle_coordinates.append([x, y])
        semicircle_offsetcoordinates.append([x_offset, y_offset])

    return semicircle_coordinates, semicircle_offsetcoordinates

# Function to generate full circle coordinates for a digit "6"
def generate_6_coordinates(radius, center_x, center_y):
    """
    Generates bigger circle coordinates for a digit "6" with given radius.
    - radius: Radius of the circular part
    - center_x, center_y: Positioning the digit "6" in the coordinate space
    """ 
    full_circle_coordinates = []
    full_circle_offsetcoordinates = []
    
    # Generate a full circle
    for theta in np.linspace(0, 2*np.pi, 40):  # Full circle
        x = center_x + (radius) * np.cos(theta) 
        x_offset = center_x + (radius+20) * np.cos(theta) 
        y_offset = center_y + (radius+20) * np.sin(theta)
        y = center_y + (radius) * np.sin(theta)
       
        full_circle_coordinates.append([x, y])
        full_circle_offsetcoordinates.append([x_offset, y_offset])
    
    return full_circle_coordinates, full_circle_offsetcoordinates 

# Initialize a canvas with zeros
canvas = np.full((1800, 500), 0)

# Define polygon coordinates for letters and numbers
e_polygon_coordinates =       [[200,110],[330,110],[330,160],[250,160],[250,210],[330,210],[330,260],[250,260],[250,310],[330,310],[330,360],[200,360]]
e_polygon_offsetcoordinates = [[180,90],[350,90], [350,180],[270,180],[270,180],[350,180],[350,280],[270,280],[270,280],[350,280],[350,380],[180,380]]

n_polygon_coordinates = [[390,110],[440,110],[440,240],[490,110],[540,110],[540,360],[490,360],[490,240],[440,360],[390,360]]
n_polygon_offsetcoordinates = [[370,90],[460,90],[460,240],[470,90],[560,90],[560,380],[470,380],[470,240],[460,380],[370,380]]

# Generate coordinates for the letter "P"
p_polygon_coordinates, p_polygon_offsetcoordinates = generate_p_coordinates(60, 650, 300)
p_stem_coordinates = [[600,360],[650,360],[650,110],[600,110]]
p_stem_offsetcoordinates = [[580,380],[670,380],[670,90],[580,90]]

m_polygon_coordinates = [[770,110],[820,110],[820,240],[840,110],[910,110],[930,240],[930,110],[980,110],[980,360],[930,360],[900,160],[850,160],[820,360],[770,360]]
m_polygon_offsetcoordinates = [[750,90],[840,90],[840,240],[840,90],[930,90],[930,240],[930,90],[1000,90],[1000,380],[910,380],[900,160],[850,160],[840,380],[750,380]]

# Generate coordinates for the digit "6"
polygon_coordinates_6, polygon_offsetcoordinates_6 = generate_6_coordinates(90,1130,200)
polygon_stem_coordinates_6 = [[1040,200],[1040,360],[1090,360],[1090,200]]
polygon_stem_offsetcoordinates_6 = [[1020,180],[1020,380],[1110,380],[1110,200]]
polygon_stem_6_circle, polygon_stem_6_circleoffset = generate_6_circle(25,1065,360)

# Generate coordinates for another "6"
polygon2_coordinates_6, polygon2_offsetcoordinates_6 = generate_6_coordinates(90,1370,200)
polygon2_stem_coordinates_6 = [[1280,200],[1280,360],[1330,360],[1330,200]]
polygon2_stem_offsetcoordinates_6 = [[1260,180],[1260,380],[1350,380],[1350,200]]
polygon2_stem_6_circle, polygon2_stem_6_circleoffset = generate_6_circle(25,1305,360)

# Generate coordinates for the digit "1"
polygon_coordinates_1 = [[1520,110],[1570,110],[1570,390],[1520,390]]
polygon_offsetcoordinates_1 = [[1500,90],[1590,90],[1590,410],[1500,410]]

# Create polygon objects for each letter and number
polygon_e = Polygon(e_polygon_offsetcoordinates, closed=True)
polygon_n = Polygon(n_polygon_offsetcoordinates, closed=True)
polygon_p = Polygon(p_polygon_offsetcoordinates, closed=True)
stem_p = Polygon(p_stem_offsetcoordinates, closed = True)
polygon_m = Polygon(m_polygon_offsetcoordinates, closed=True)

polygon_6 = Polygon(polygon_offsetcoordinates_6, closed=True)
polygon_6_stem = Polygon(polygon_stem_offsetcoordinates_6, closed=True)
polygon_6_stem_circle = Polygon(polygon_stem_6_circleoffset, closed=True)

polygon2_6 = Polygon(polygon2_offsetcoordinates_6, closed=True)
polygon2_6_stem = Polygon(polygon2_stem_offsetcoordinates_6, closed=True)
polygon2_6_stem_circle = Polygon(polygon2_stem_6_circleoffset, closed=True)

polygon_1 = Polygon(polygon_offsetcoordinates_1, closed=True)

# Set boundaries of the canvas
canvas[0:5,:], canvas[1795:,:], canvas[:,0:5], canvas[:,495:] = 1,1,1,1

# List of polygons representing obstacles
obstacles = [
    polygon_e, polygon_n, polygon_p, stem_p, polygon_m,
    polygon_6, polygon_6_stem, polygon_6_stem_circle,
    polygon2_6, polygon2_6_stem, polygon2_6_stem_circle,
    polygon_1
]

# Iterate through each pixel on the canvas and mark obstacles
for y in range(500):
    for x in range(1800):
        for theta in range(360):
            for obstacle in obstacles:
                if obstacle.contains_point((x, y,theta)):
                    canvas[x, y,theta] = 1
                    break  # No need to check other obstacles once marked

# Function to convert coordinates for pygame
def coords_pygame(coords, height):
    """
    Converts coordinates to pygame format by flipping the y-axis.
    """
    return (coords[0], height - coords[1])

def create_map(visit, backtrack, start, Goal):
    """
    Visualizes the path on a map using pygame.
    """
    pygame.init()
    size = [1800, 500]
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("BFS - Amogha Sunil")

    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        screen.fill("white")
        
        # List of polygons to draw with interesting colors
        polygons = [
            (e_polygon_coordinates, (255, 0, 0)),  # Red
            (n_polygon_coordinates, (0, 0, 255)),  # Blue
            (p_polygon_coordinates, (0, 255, 0)),  # Green
            (p_stem_coordinates, (0, 255, 0)),  # Green
            (m_polygon_coordinates, (255, 165, 0)),  # Orange
            (polygon_coordinates_6, (128, 0, 128)),  # Purple
            (polygon_stem_coordinates_6, (128, 0, 128)),  # Purple
            (polygon_stem_6_circle, (128, 0, 128)),  # Purple
            (polygon2_coordinates_6, (75, 0, 130)),  # Deep Purple
            (polygon2_stem_coordinates_6, (75, 0, 130)),  # Deep Purple
            (polygon2_stem_6_circle, (75, 0, 130)),  # Deep Purple
            (polygon_coordinates_1, (255, 255, 0))  # Yellow
        ]
        
        # Convert and draw polygons
        for polygon, color in polygons:
            pygame_polygon = [coords_pygame(point, 500) for point in polygon]
            pygame.draw.polygon(screen, color, pygame_polygon, 0)
        
        # Draw boundary
        pygame.draw.rect(screen, "blue", [0,0, 1800, 500], 5)
        
        # Draw visited nodes
        n = 0
        for j in visit:
            n += 1
            pygame.draw.circle(screen, (50, 137, 131), coords_pygame(j, 500), 1)
            if n % 50 == 0:
                pygame.display.update()
        
        # Draw start and goal points
        pygame.draw.circle(screen, (0, 255, 0), coords_pygame(start, 500), -3)
        pygame.draw.circle(screen, (0, 255, 0), coords_pygame(Goal, 500), -3)
        
        # Draw the path
        for i in backtrack:
            pygame.draw.circle(screen, (255, 255, 0), coords_pygame(i, 500), 1)
            pygame.display.update()
        pygame.draw.circle(screen, (255, 255, 0), coords_pygame(start, 500), 1)
        pygame.draw.circle(screen, (255, 255, 0), coords_pygame(Goal, 500), 1)
        
        done = True
    pygame.time.wait(5000)
    pygame.quit()

# Function to check if a point is an obstacle  (OK)
def check_obstacles(coordinates):
    """
    Checks if a point lies on an obstacle.
    """
    if canvas[coordinates[0], coordinates[1],coordinates[2]] == 1:
        return False
    return True

# Function to input start or goal position
# modify the inputs to include theta

def input_start(prompt):   
    """
    Prompts user to input a start or goal position.
    """
    
    while True:
        print("Enter", prompt, "node (x,y,θ) (x between 5 and 594, y between 5 and 244, θ in multiples of 30°) ")
        print("Sample Input: 10,10,0")
        input_str = input()
        A = [int(i) for i in input_str.split(',')]
        A_1 = (A[0], A[1],A[2])
        if not (0 <= A[0] < 1800 and 0 <= A[1] < 500 and 0 <= A[2] < 360):  # Check if the input is within bounds (0 to 360 or -180 to 180)
            print("Enter valid input (x between 5 and 1794, y between 5 and 494)")
        elif not check_obstacles(A_1):
            print("The entered input lies on the obstacles or is not valid, please try again")
        else:
            return A_1

def a_star():
    start_time = time.time()  # Calculate the time required to run
    
    # Initialize a queue with the start position
    Q = Queue()
    Q.put(Start)
    
    # Initialize a set to keep track of visited nodes
    visit = OrderedSet()
    visit.add(Start)
    
    # Initialize a dictionary to keep track of the path
    Path = {}
    
    while not Q.empty():
        current_node = Q.get()
        
        # Check if the goal is reached
        if current_node == goal:
            print('success')
            # Generate and print the path
            Backtrack = generate_path_bfs(Path, Start, goal)
            print(Backtrack)
            print('-----------')
            end_time = time.time()
            path_time = end_time - start_time
            print('Time to calculate path:', path_time, 'seconds')
            # Visualize the path using pygame
            create_map(visit, Backtrack, Start, goal)
            break
        
        # # Explore all possible moves from the current node
        # movements = [        
        #     (0, 1),  # Up
        #     (0, -1),  # Down
        #     (-1, 0),  # Left
        #     (1, 0),  # Right
        #     (-1, 1),  # Up-left
        #     (1, 1),  # Up-right
        #     (-1, -1),  # Down-left
        #     (1, -1)  # Down-right
        # ]
        
        # Define the 5 actions
        step_size = 1  # Define the step size (can be adjusted or taken as input)
        actions = [
            # Forward
            (step_size * math.cos(math.radians(current_node[2])), 
            step_size * math.sin(math.radians(current_node[2])), 
            0),
            
            # Forward Left (30° left)
            (step_size * math.cos(math.radians(current_node[2] + 30)), 
            step_size * math.sin(math.radians(current_node[2] + 30)), 
            30),
            
            # Forward Right (30° right)
            (step_size * math.cos(math.radians(current_node[2] - 30)), 
            step_size * math.sin(math.radians(current_node[2] - 30)), 
            -30),
            
            # Left Turn (60° left)
            (step_size * math.cos(math.radians(current_node[2] + 60)), 
            step_size * math.sin(math.radians(current_node[2] + 60)), 
            60),
            
            # Right Turn (60° right)
            (step_size * math.cos(math.radians(current_node[2] - 60)), 
            step_size * math.sin(math.radians(current_node[2] - 60)), 
            -60)
        ]

        for dx, dy,dtheta in actions:
            new_x, new_y,new_theta = current_node[0] + dx, current_node[1] + dy,current_node[2]+dtheta
            # new_theta = new_theta % 360 
            
            # Check if the new position is within bounds and not an obstacle
            if (0 <= new_x < 250) and (0 <= new_y < 600 and 0<=new_theta <360) and check_obstacles((new_x, new_y,new_theta)) and (new_x, new_y,new_theta) not in visit:
                Q.put((new_x, new_y,new_theta))
                visit.add((new_x, new_y,new_theta))
                Path[(new_x, new_y,new_theta)] = current_node


def generate_path_bfs(path, start, goal):
    """
    Reconstructs the path from the goal to the start using backtracking.
    """
    backtrack = []
    key = path.get(goal)
    backtrack.append(goal)
    backtrack.append(key)
    while (key != start):
        key = path.get(key)
        backtrack.append(key)
    backtrack.reverse()
    return backtrack


visit = OrderedSet()
touch = {}
Path = {}
Start = input_start('Start Position')
goal = input_start('Goal Position')
print(Start, goal)
a_star()