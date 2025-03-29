# Pathfinding A star Algorithm: 
This repository contains implementations of A* algorithm for a circular robot.

## Overview 

The A* (A-star) algorithm is a widely used pathfinding and graph traversal algorithm that finds the shortest path from a start node to a goal node. It is commonly used in robotics, game development, and AI applications.

How It Works

A* combines the advantages of:

Dijkstra’s Algorithm (which finds the shortest path by exploring all possible routes)

Greedy Best-First Search (which prioritizes paths that seem promising based on a heuristic function)


It uses the following cost function for each node:

f(n) = g(n) + h(n)

g(n) = actual cost from the start node to node n

h(n) = heuristic estimate of the cost from node n to the goal

f(n) = estimated total cost of the path through n



## Libraries Used

- **NumPy**: For numerical computations.
- **OpenCV**: For visualizing the path.
- **PriorityQueue**
  
  
## How to Run

1. **Install Required Libraries**:

2. **Run A Star algorithm**:
- python3 Project3_Amogha_Sagar_Shreya.py
- Run the script and follow the prompts to input start and goal positions and orientation.

## Basic Information

- **Obstacles**: Defined by polygons representing letters and numbers.
- **Path Visualization**: Uses opencv to display the shortest path.

video demo:
