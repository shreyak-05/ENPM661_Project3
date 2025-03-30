This repository contains an implementation of the A* (A-star) algorithm for pathfinding with a circular robot in a 2D workspace with obstacles.

## Overview

The A* (A-star) algorithm is a widely used pathfinding and graph traversal algorithm that finds the shortest path from a start node to a goal node. It is commonly used in robotics, game development, and AI applications. This implementation is tailored for a circular robot, considering its physical dimensions and clearance requirements.

## How It Works

A* combines the advantages of:

* **Dijkstra’s Algorithm:** Finds the shortest path by exploring all possible routes.
* **Greedy Best-First Search:** Prioritizes paths that seem promising based on a heuristic function.

It uses the following cost function for each node:

   `f(n) = g(n) + h(n)`

Where:

* `g(n)` = Actual cost from the start node to node n.
* `h(n)` = Heuristic estimate of the cost from node n to the goal.
* `f(n)` = Estimated total cost of the path through n.

In this implementation, the heuristic function `h(n)` is the Euclidean distance between the current node and the goal node.

## Features

* **Circular Robot Navigation:** Considers the robot's radius and clearance to avoid collisions.
* **Obstacle Avoidance:** Handles complex obstacle shapes defined by polygons representing letters and numbers.
* **Rotational Planning:** Includes robot orientation (`theta`) in the state space for more accurate path planning.
* **Path Visualization:** Uses OpenCV to display the shortest path found.
* **User Input:** Prompts the user to input start and goal positions and orientations.

## Libraries Used

* **NumPy:** For numerical computations and array manipulations.
* **OpenCV (cv2):** For visualizing the path and obstacle environments.
* **PriorityQueue (from queue):** For efficient node selection in the A* algorithm.
* **Time:** For measuring the execution time of the algorithm.

## How to Run

1.  **Install Required Libraries:**

    ```bash
    pip install numpy opencv-python
    ```

2.  **Run the A* Algorithm:**

    ```bash
    python3 Project3_Amogha_Sagar_Shreya.py
    ```

3.  **Input Start and Goal Positions:**

    * The script will prompt you to enter the start and goal coordinates (X, Y) and orientation (Theta).
    * Coordinates should be within the canvas dimensions (600x250).
    * Theta should be a multiple of 30, between 0 and 360.
    * The script will validate the inputs to ensure they are within the workspace and not within obstacles or clearance zones.
    * You will also be prompted for the step size, and clearance amount.

## Basic Information

* **Obstacles:** The workspace contains obstacles shaped as letters and numbers, defined as polygons.
* **Path Visualization:** The script uses OpenCV to display the workspace, obstacles, and the calculated shortest path.
* **Clearance:** The robot has a clearance zone around it, which is considered during path planning to prevent collisions.
* **Theta:** The robot orientation is a parameter in the search space, allowing for rotational planning.

## Team Members

* Amogha
    * Directory ID: [Your Directory ID]
    * UID: [Your UID]
* Sagar
    * Directory ID: [Sagar's Directory ID]
    * UID: [Sagar's UID]
* Shreya
    * Directory ID: shreya05
    * UID: 121166647
