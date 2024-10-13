import serial
import time
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib import colors
import heapq

# --------------------------- Configuration --------------------------- #

SERIAL_PORT = "COM5"
BAUD_RATE = 9600
SERIAL_TIMEOUT = 1  # in seconds

OBSTACLE_THRESHOLD = 50.0  # Distance in cm to stop for obstacles
GRID_SIZE = 40
CELL_SIZE = 5.0
MAX_DISTANCE = 200.0

ORIGIN_X = GRID_SIZE // 2
ORIGIN_Y = GRID_SIZE // 2

grid = np.zeros((GRID_SIZE, GRID_SIZE))

waypoints = [(5, 5), (10, 10), (20, 15), (30, 35)]  # Predefined waypoints
current_waypoint_index = 0

# Robot's current position (start at origin)
robot_position = (ORIGIN_X, ORIGIN_Y)

# Keep track of obstacle positions
obstacle_positions = set()  # Set to store obstacle positions

# --------------------------- Initialization --------------------------- #

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
    print(f"Connected to Arduino on port {SERIAL_PORT}")
except serial.SerialException:
    print(f"Failed to connect to Arduino on port {SERIAL_PORT}")
    exit()

time.sleep(2)

# Initialize plot
fig, ax = plt.subplots()
cmap = colors.ListedColormap(["grey", "white", "black"])
bounds = [0, 1, 2, 3]
norm = colors.BoundaryNorm(bounds, cmap.N)

img = ax.imshow(grid, cmap=cmap, norm=norm, origin="lower")
ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)
ax.set_xlabel("X (cells)")
ax.set_ylabel("Y (cells)")
ax.set_title("Dynamic Grid-Based Local Map")

(robot_marker,) = ax.plot(ORIGIN_X, ORIGIN_Y, "bo")  # Blue dot representing robot

# --------------------------- Data Parsing --------------------------- #


def parse_serial_data(line):
    angle = None
    distance = None
    state = None
    alert = False

    if "Robot State:" in line:
        state = line.split("Robot State:")[1].strip()
    elif "[ALERT]" in line:
        alert = True
    elif "Angle:" in line and "Distance:" in line:
        try:
            parts = line.split("|")
            if len(parts) != 2:
                return angle, distance, state, alert

            angle_part = parts[0].strip().split(":")[1].strip().replace("°", "")
            angle = float(angle_part)

            distance_part = (
                parts[1].strip().split(":")[1].strip().replace("cm", "").strip()
            )
            distance = float(distance_part)
        except (IndexError, ValueError):
            pass

    return angle, distance, state, alert


# --------------------------- Grid Mapping --------------------------- #


def update_grid_relative_to_robot(angle, distance):
    """
    Updates the grid based on the robot's current position and ultrasonic sensor readings.
    The sensor detects obstacles in front of the robot.
    """
    global robot_position, obstacle_positions

    if distance > MAX_DISTANCE:
        distance = MAX_DISTANCE

    # Convert the robot-relative distance to grid coordinates
    radians = math.radians(angle)
    x_cm = distance * math.cos(radians)
    y_cm = distance * math.sin(radians)

    # Add the detected distance to the robot's current position
    x_cell = int(round(robot_position[0] + (x_cm / CELL_SIZE)))
    y_cell = int(round(robot_position[1] + (y_cm / CELL_SIZE)))

    # Ensure the detected obstacle is within grid boundaries
    if 0 <= x_cell < GRID_SIZE and 0 <= y_cell < GRID_SIZE:
        # Check if the obstacle is detected and mark the cell accordingly
        if distance < OBSTACLE_THRESHOLD:
            grid[y_cell][x_cell] = 2  # Occupied
            obstacle_positions.add((x_cell, y_cell))  # Add to the obstacle set
        else:
            # Clear the cell if no obstacle is detected
            if (x_cell, y_cell) in obstacle_positions:
                grid[y_cell][x_cell] = 1  # Free space
                obstacle_positions.discard((x_cell, y_cell))  # Remove from obstacle set


# --------------------------- Path Planning (A*) --------------------------- #


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(grid, start, goal):
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while frontier:
        current = heapq.heappop(frontier)[1]

        if current == goal:
            break

        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in neighbors:
            next_node = (current[0] + dx, current[1] + dy)

            if 0 <= next_node[0] < GRID_SIZE and 0 <= next_node[1] < GRID_SIZE:
                if grid[next_node[1]][next_node[0]] != 2:  # Avoid obstacles
                    new_cost = cost_so_far[current] + 1

                    if (
                        next_node not in cost_so_far
                        or new_cost < cost_so_far[next_node]
                    ):
                        cost_so_far[next_node] = new_cost
                        priority = new_cost + heuristic(goal, next_node)
                        heapq.heappush(frontier, (priority, next_node))
                        came_from[next_node] = current

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]

    path.reverse()
    return path


# --------------------------- Move the Robot --------------------------- #


def move_robot_along_path(path):
    global robot_position

    if len(path) > 1:  # If there are still steps to take
        next_step = path[1]  # Move to the next cell on the path
        robot_position = next_step  # Update the robot's position
        print(f"Robot moved to {robot_position}")
    return robot_position


# --------------------------- Stop When Obstacle Detected --------------------------- #


def check_for_obstacle(distance):
    """
    Checks if there's an obstacle within the OBSTACLE_THRESHOLD distance.
    :param distance: Distance from the ultrasonic sensor (in cm).
    :return: True if an obstacle is detected, otherwise False.
    """
    if distance < OBSTACLE_THRESHOLD:
        print(f"[STOP] Obstacle detected within {distance} cm!")
        return True
    return False


def update_grid_and_path(distance):
    global current_waypoint_index, waypoints, robot_position

    # Stop the robot if an obstacle is detected
    if check_for_obstacle(distance):
        return  # Stop the robot from moving if there's an obstacle

    if current_waypoint_index < len(waypoints):
        goal = waypoints[current_waypoint_index]  # Get the current waypoint
        path = a_star(grid, robot_position, goal)  # Generate the path using A*

        if path and robot_position != goal:
            move_robot_along_path(path)  # Move the robot along the path
        else:
            current_waypoint_index += 1  # Move to the next waypoint once reached
            print(f"Reached waypoint {current_waypoint_index}: {goal}")


# --------------------------- Plot Updating --------------------------- #


def update_plot(frame):
    if ser.in_waiting:
        try:
            line = ser.readline().decode("utf-8").rstrip()
            if line:
                angle, distance, state, alert = parse_serial_data(line)

                if angle is not None and distance is not None:
                    update_grid_relative_to_robot(
                        angle, distance
                    )  # Move sensor with the robot
                    update_grid_and_path(distance)  # Move robot along the path

                    img.set_data(grid)
                    robot_marker.set_data(
                        robot_position[0], robot_position[1]
                    )  # Update blue dot
                    fig.canvas.draw_idle()

        except UnicodeDecodeError:
            pass

    return img, robot_marker


# --------------------------- Animation --------------------------- #

ani = animation.FuncAnimation(fig, update_plot, interval=100)

plt.show()

ser.close()
print("Serial connection closed.")
