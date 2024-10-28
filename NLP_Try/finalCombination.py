import pygame
import math
import sys
import serial
import threading
import logging
import time
import queue
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from gtts import gTTS
import os
import subprocess
from transformers import pipeline
from pydub import AudioSegment
from pydub.playback import play
import uuid
import webbrowser

# Initialize thread-safe queues
command_queue = queue.Queue()
response_queue = queue.Queue()
prompt_queue = queue.Queue()  # Queue for prompts

# -------------------- Constants ---------------------#

# Pygame Initialization
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
BUTTON_AREA_HEIGHT = 200  # Space for buttons and prompt
TOTAL_HEIGHT = HEIGHT + BUTTON_AREA_HEIGHT  # Total screen height with buttons area

# Color definitions
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
DARK_GRAY = (169, 169, 169)
LIGHT_GRAY = (211, 211, 211)

# Car properties
CAR_IMAGE_PATH = (
    "navigationTry/2d-super-car-top-view.png"  # Ensure this path is correct
)
CAR_SIZE = (80, 40)  # Reduced from (100, 50)
CAR_SPEED = 2
CAR_ROTATION_SPEED = 5

# Sensor properties
NUM_SENSORS = 3
SENSOR_LENGTH = 50  # Adjusted sensor length
SENSOR_ANGLES = [-30, 0, 30]  # Angles for left, front, right sensors

# Waypoint properties
WAYPOINT_THRESHOLD = 20  # Distance to consider waypoint as reached

# Frame rate
FPS = 60

# Serial communication settings
SERIAL_PORT = "COM5"  # Update this as per your system
BAUD_RATE = 115200  # Updated to match Arduino's baud rate

# -------------------- Logging Configuration ---------------------#

# Configure the logging module
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to console
    ],
)

# Create a logger object
logger = logging.getLogger(__name__)

# -------------------- Flask App Setup ---------------------#

# Initialize the zero-shot classification pipeline
nlp = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    clean_up_tokenization_spaces=True,
)

# Updated labels to include navigation commands
labels = [
    "open_browser",
    "open_notepad",
    "play_music",
    "turn_on_lights",
    "kill",
    "go_to_M215",
    "go_to_M216",
    "go_to_M217",
    "none",
]

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  # Enable CORS for all routes if frontend is on a different origin


def open_application(command):
    """Prepares the appropriate response based on the command."""
    response = ""
    if command == "open_browser":
        logger.info("Attempting to open the browser...")
        try:
            subprocess.Popen(["start", "chrome"], shell=True)
            response = "Opening the browser."
        except Exception as e:
            logger.error(f"Failed to open browser: {e}")
            response = "Failed to open the browser."
    elif command == "open_notepad":
        logger.info("Attempting to open Notepad...")
        try:
            subprocess.Popen(["notepad.exe"])
            response = "Opening Notepad."
        except Exception as e:
            logger.error(f"Failed to open Notepad: {e}")
            response = "Failed to open Notepad."
    elif command == "play_music":
        logger.info("Playing music.")
        response = "Playing music."
        # You can add functionality here to play music
    elif command == "turn_on_lights":
        logger.info("Turning on the lights.")
        response = "Turning on the lights."
        # Add smart light control code here if you want
    elif command == "kill":
        logger.info("Stopping the program. Goodbye!")
        response = "Stopping the program. Goodbye!"
        # Optionally, you can stop the Flask server or exit
        # os._exit(0)  # Be cautious with using os._exit
    elif command.startswith("go_to_"):
        # Extract room number and enqueue navigation command
        room = command.split("_")[2]
        response = f"Taking you to room {room}."
        command_queue.put(command)  # Place navigation command in the queue
    elif command == "none":
        response = "I didn't understand the command."
    else:
        response = "I didn't understand the command."

    # Enqueue the response for Pygame to handle TTS
    response_queue.put(response)
    return jsonify({"response": response})


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/command", methods=["POST"])
def handle_command():
    """Handles the POST request from the frontend, processes the command, and sends back a response."""
    data = request.json
    logger.debug(f"Received data: {data}")  # Debugging statement
    command_text = data.get("text", "").strip()

    if command_text:
        logger.debug(f"Command received: {command_text}")  # Debugging statement
        result = nlp(command_text, candidate_labels=labels, multi_label=False)
        logger.debug(f"Model result: {result}")  # Debugging statement
        predicted_label = result["labels"][0]
        confidence = result["scores"][0]

        logger.info(
            f"Predicted label: {predicted_label} with confidence {confidence}"
        )  # Debugging

        # If confidence is above 0.5, execute the application
        if confidence > 0.5:
            return open_application(predicted_label)
        else:
            response = "I'm not sure what you meant. Can you try again?"
            response_queue.put(response)
            return jsonify({"response": response})

    return jsonify({"response": "No command received."})


@app.route("/get_prompt", methods=["GET"])
def get_prompt():
    """Return the prompt message if available."""
    try:
        prompt = prompt_queue.get_nowait()
        return jsonify({"prompt": prompt})
    except queue.Empty:
        return jsonify({"prompt": None})


@app.route("/post_choice", methods=["POST"])
def post_choice():
    """Receive the user's choice and put it into command_queue."""
    data = request.json
    choice = data.get("choice")
    if choice:
        command_queue.put(choice)
        return jsonify({"status": "success"})
    else:
        return jsonify({"status": "error", "message": "No choice provided."}), 400


def open_browser_after_delay(url, delay=1):
    """Waits for a specified delay and then opens the browser."""
    time.sleep(delay)
    webbrowser.open(url)


# -------------------- Wall Class ---------------------#


class Wall:
    """Represents a wall in the environment."""

    def __init__(self, rect):
        """
        Initialize a Wall object.

        Args:
            rect (pygame.Rect): The rectangle defining the wall's position and size.
        """
        self.rect = rect
        self.mask = self.create_mask()

    def create_mask(self):
        """
        Create a mask for collision detection.

        Returns:
            pygame.Mask: The mask representing the wall.
        """
        wall_surface = pygame.Surface(
            (self.rect.width, self.rect.height), pygame.SRCALPHA
        )
        wall_surface.fill(BLACK)
        return pygame.mask.from_surface(wall_surface)

    def draw(self, surface):
        """
        Draw the wall on the given surface.

        Args:
            surface (pygame.Surface): The surface to draw the wall on.
        """
        pygame.draw.rect(surface, BLACK, self.rect)


# -------------------- CarRobot Class ---------------------#


class CarRobot:
    """Represents the autonomous car."""

    def __init__(self, x, y, waypoints, waypoint_names, walls, prompt_queue):
        """
        Initialize the CarRobot.

        Args:
            x (float): Initial x-coordinate.
            y (float): Initial y-coordinate.
            waypoints (list): List of waypoint tuples.
            waypoint_names (list): List of names for each waypoint.
            walls (list): List of Wall objects for collision detection.
            prompt_queue (queue.Queue): Queue to send prompts to the frontend.
        """
        self.start_x = x
        self.start_y = y
        self.x = x
        self.y = y
        self.angle = 0  # Facing right initially
        self.speed = CAR_SPEED
        self.waypoints = waypoints
        self.waypoint_names = waypoint_names
        self.current_target = None  # Current target position
        self.current_location_name = "Start"  # Current location name
        self.destination_name = None  # Destination location name
        self.moving = False  # Car should start stationary
        self.threshold = WAYPOINT_THRESHOLD
        self.walls = walls
        self.load_image()
        self.state_reason = "Waiting for waypoint"  # Reason for stop or move
        self.is_returning_to_start = False  # Flag to track returning to start
        self.prompt_queue = prompt_queue  # Queue to communicate with frontend
        self.sensors = []
        self.create_sensors()
        self.path = []  # List of waypoints to follow
        self.arduino_obstacle_detected = False  # Flag for Arduino obstacle detection

    def load_image(self):
        """Load and scale the car image."""
        try:
            self.original_image = pygame.image.load(CAR_IMAGE_PATH)
            self.original_image = pygame.transform.scale(self.original_image, CAR_SIZE)
            logger.debug("Car image loaded successfully.")
        except pygame.error as e:
            logger.error(f"Failed to load car image: {e}")
            sys.exit()

    def create_sensors(self):
        """Initialize the sensor positions based on current angle."""
        self.sensors = []
        for angle_offset in SENSOR_ANGLES:
            sensor_angle = (self.angle + angle_offset) % 360
            sensor_end_x = self.x + SENSOR_LENGTH * math.cos(math.radians(sensor_angle))
            sensor_end_y = self.y - SENSOR_LENGTH * math.sin(math.radians(sensor_angle))
            self.sensors.append((sensor_angle, (sensor_end_x, sensor_end_y)))

    def update_sensors(self):
        """Update sensor positions."""
        self.create_sensors()

    def check_sensors(self):
        """Check sensors for obstacle detection."""
        sensor_data = []
        for sensor_angle, (sensor_end_x, sensor_end_y) in self.sensors:
            # Create a line from (self.x, self.y) to (sensor_end_x, sensor_end_y)
            sensor_line = ((self.x, self.y), (sensor_end_x, sensor_end_y))
            obstacle_detected = False
            for wall in self.walls:
                if self.line_rect_intersect(sensor_line, wall.rect):
                    obstacle_detected = True
                    break
            sensor_data.append((sensor_angle, obstacle_detected))
        return sensor_data

    def line_rect_intersect(self, line, rect):
        """Check if a line intersects with a rectangle."""
        rect_lines = [
            ((rect.left, rect.top), (rect.right, rect.top)),  # Top
            ((rect.right, rect.top), (rect.right, rect.bottom)),  # Right
            ((rect.right, rect.bottom), (rect.left, rect.bottom)),  # Bottom
            ((rect.left, rect.bottom), (rect.left, rect.top)),  # Left
        ]
        for rect_line in rect_lines:
            if self.line_line_intersect(line, rect_line):
                return True
        return False

    def line_line_intersect(self, line1, line2):
        """Check if two lines intersect."""
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2

        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denom == 0:
            return False  # Lines are parallel
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            return True
        else:
            return False

    def get_target_angle(self):
        """
        Calculate the angle towards the current target point.

        Returns:
            float: The angle in degrees towards the target point.
        """
        if not self.current_target:
            return self.angle  # No target, return current angle

        target_x, target_y = self.current_target
        dx = target_x - self.x
        dy = self.y - target_y  # Inverted y-axis for Pygame
        angle = math.degrees(math.atan2(dy, dx))
        return angle % 360

    def rotate_towards_target(self, target_angle):
        """
        Rotate the car towards the target angle.

        Args:
            target_angle (float): The desired angle in degrees.
        """
        angle_diff = (target_angle - self.angle + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360

        if abs(angle_diff) < CAR_ROTATION_SPEED:
            self.angle = target_angle
        elif angle_diff > 0:
            self.angle += CAR_ROTATION_SPEED
        else:
            self.angle -= CAR_ROTATION_SPEED

        self.angle %= 360
        self.update_sensors()

    def move_forward(self):
        """Move the car forward based on its speed and angle."""
        new_x = self.x + self.speed * math.cos(math.radians(self.angle))
        new_y = self.y - self.speed * math.sin(math.radians(self.angle))
        if not self.check_collision(new_x, new_y):
            self.x = new_x
            self.y = new_y
        else:
            logger.warning("Collision detected! Movement blocked.")

    def check_point_reached(self):
        """Check if the current target point has been reached."""
        if not self.current_target:
            return False

        target_x, target_y = self.current_target
        distance = math.hypot(target_x - self.x, target_y - self.y)
        if distance < self.threshold:
            logger.info(f"Reached point ({target_x}, {target_y})")
            # Update current location name if we have a name for this waypoint
            waypoint_name = self.get_waypoint_name(self.current_target)
            if waypoint_name:
                self.current_location_name = waypoint_name

            if self.path:
                # Move to the next point in the path
                self.current_target = self.path.pop(0)
                self.state_reason = f"Moving towards waypoint ({self.current_target[0]}, {self.current_target[1]})"
            else:
                # Reached final destination
                self.current_location_name = self.destination_name
                self.current_target = None
                self.moving = False  # Stop moving after reaching the final point
                destination_name = self.current_location_name
                if not self.is_returning_to_start:
                    # Send prompt to frontend
                    prompt_message = f"Reached {destination_name}"
                    self.prompt_queue.put(prompt_message)
                    self.state_reason = "Awaiting user choice"
                else:
                    # If returning to start, reset the flag and set reason
                    self.is_returning_to_start = False
                    self.state_reason = "At Start Point"
            return True
        return False

    def get_waypoint_name(self, position):
        """Get the name of the waypoint given its position."""
        for name, pos in zip(self.waypoint_names, self.waypoints):
            if position == pos:
                return name
        return None

    def update_mask(self):
        """
        Update the car's mask for collision detection after rotation.

        Returns:
            tuple: Rotated image and its corresponding rect.
        """
        rotated_image = pygame.transform.rotate(self.original_image, self.angle)
        rotated_rect = rotated_image.get_rect(center=(self.x, self.y))
        self.car_mask = pygame.mask.from_surface(rotated_image)
        return rotated_image, rotated_rect

    def check_collision(self, new_x, new_y):
        """
        Check for collisions at the new position.

        Args:
            new_x (float): Proposed new x-coordinate.
            new_y (float): Proposed new y-coordinate.

        Returns:
            bool: True if collision occurs, False otherwise.
        """
        original_x, original_y = self.x, self.y
        self.x, self.y = new_x, new_y
        rotated_image, rect = self.update_mask()
        car_mask = pygame.mask.from_surface(rotated_image)

        collision = False
        for wall in self.walls:
            offset = (int(wall.rect.x - rect.x), int(wall.rect.y - rect.y))
            overlap = car_mask.overlap(wall.mask, offset)
            if overlap:
                logger.warning(f"Collision with wall at position: {wall.rect}")
                collision = True
                break

        if collision:
            self.x, self.y = original_x, original_y
        return collision

    def set_target(self, target_point, destination_name):
        """Set the target point for the car to move towards."""
        self.destination_name = destination_name
        self.moving = True
        self.state_reason = f"Moving towards {destination_name}"
        self.update_sensors()

        path_key = (self.current_location_name, destination_name)
        if path_key in self.waypoint_paths:
            # Get the path of positions
            self.path = [
                self.waypoint_dict[wp_name] for wp_name in self.waypoint_paths[path_key]
            ]
            self.current_target = self.path.pop(0)
        else:
            self.current_target = target_point
            self.path = []

    def return_to_start(self):
        """Move the car back to the starting position."""
        self.is_returning_to_start = True
        self.set_target((self.start_x, self.start_y), "Start")

    def update(self):
        """Update the car's state by rotating and moving towards the target."""
        if self.moving and self.current_target:
            # Check if Arduino has detected an obstacle
            if self.arduino_obstacle_detected:
                self.moving = False
                self.state_reason = "Obstacle detected by Arduino"
                return

            # First, check sensors for obstacles
            sensor_data = self.check_sensors()
            obstacles = [detected for angle, detected in sensor_data if detected]

            # If close to the target, disable obstacle avoidance
            target_distance = math.hypot(
                self.current_target[0] - self.x, self.current_target[1] - self.y
            )
            if target_distance < self.threshold * 2:
                obstacles = []

            if obstacles:
                # Obstacle detected ahead, adjust path
                left_sensor = sensor_data[0][1]
                front_sensor = sensor_data[1][1]
                right_sensor = sensor_data[2][1]

                if front_sensor:
                    if not left_sensor and right_sensor:
                        self.angle = (self.angle + CAR_ROTATION_SPEED) % 360
                        self.state_reason = "Obstacle ahead - Turning left"
                    elif not right_sensor and left_sensor:
                        self.angle = (self.angle - CAR_ROTATION_SPEED) % 360
                        self.state_reason = "Obstacle ahead - Turning right"
                    elif not left_sensor and not right_sensor:
                        self.angle = (self.angle + CAR_ROTATION_SPEED) % 360
                        self.state_reason = "Obstacle ahead - Turning left"
                    else:
                        self.moving = False
                        self.state_reason = "Obstacle ahead, cannot avoid"
                    self.update_sensors()
                else:
                    # Side obstacles, proceed forward
                    self.move_forward()
                    self.check_point_reached()
            else:
                # No obstacle ahead, proceed towards target
                target_angle = self.get_target_angle()
                self.rotate_towards_target(target_angle)
                self.move_forward()
                self.check_point_reached()
        else:
            if self.arduino_obstacle_detected:
                self.state_reason = "Obstacle detected by Arduino"
            else:
                self.state_reason = "Stopped"

    def draw_status(self, surface):
        """Draw the current status and reason for stop/move on the screen."""
        font = pygame.font.SysFont(None, 24)
        status = "MOVING" if self.moving else "STOPPED"
        status_text = font.render(f"Robot Status: {status}", True, BLACK)
        reason_text = font.render(f"Reason: {self.state_reason}", True, BLACK)

        # Draw status and reason
        surface.blit(status_text, (10, 10))
        surface.blit(reason_text, (10, 40))

    def draw(self, surface):
        """Render the car and waypoints on the screen."""
        rotated_image, rect = self.update_mask()
        surface.blit(rotated_image, rect.topleft)

        # Draw sensors with color indicating obstacle detection
        sensor_data = self.check_sensors()
        for (sensor_angle, (sensor_end_x, sensor_end_y)), (_, obstacle_detected) in zip(
            self.sensors, sensor_data
        ):
            # If Arduino has detected an obstacle, override the sensor color
            if self.arduino_obstacle_detected:
                color = RED
            else:
                color = RED if obstacle_detected else GREEN
            pygame.draw.line(
                surface, color, (self.x, self.y), (sensor_end_x, sensor_end_y), 2
            )
            pygame.draw.circle(
                surface, color, (int(sensor_end_x), int(sensor_end_y)), 3
            )

        # Draw waypoints
        for idx, (wp_x, wp_y) in enumerate(self.waypoints):
            color = (
                GREEN
                if self.waypoint_names[idx] == self.current_location_name
                else BLUE
            )
            pygame.draw.circle(surface, color, (int(wp_x), int(wp_y)), 8)
            font = pygame.font.SysFont(None, 24)
            img = font.render(self.waypoint_names[idx], True, BLACK)
            surface.blit(img, (wp_x + 10, wp_y - 10))

    # Define the predefined paths between waypoints
    waypoint_paths = {
        ("Start", "M215"): ["M215"],
        ("M215", "M216"): ["M216"],
        ("M216", "M217"): ["M217"],
        ("M217", "Start"): ["Start"],
        ("Start", "M216"): ["M215", "M216"],
        ("Start", "M217"): ["M215", "M216", "M217"],
        ("M215", "M217"): ["M216", "M217"],
        ("M216", "M215"): ["M215"],
        ("M216", "Start"): ["M215", "Start"],
        ("M217", "M216"): ["M216"],
        ("M217", "M215"): ["M216", "M215"],
        ("M215", "Start"): ["Start"],
        # Add more paths as needed
    }

    # Placeholder for the waypoint dictionary (to be set in Game class)
    waypoint_dict = {}


# -------------------- SerialReader Class ---------------------#


class SerialReader(threading.Thread):
    """Handles serial communication with the Arduino."""

    def __init__(self, serial_port, baud_rate, car, game):
        """
        Initialize the SerialReader thread.

        Args:
            serial_port (str): The serial port to connect to.
            baud_rate (int): The baud rate for communication.
            car (CarRobot): The car object to control based on serial data.
            game (Game): Reference to the Game object for sending commands.
        """
        super().__init__()
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.running = True
        self.ser = None
        self.car = car  # Car instance to control
        self.game = game  # Reference to Game for sending commands
        self.state = "STOPPED"  # Initialize to STOPPED
        self.lock = threading.Lock()

    def run(self):
        """Run the thread to continuously read from the serial port."""
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            logger.info(
                f"Connected to Arduino on {self.serial_port} at {self.baud_rate} baud."
            )
            while self.running:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode("utf-8").strip()
                    if line.startswith("<STATE>") and line.endswith("</STATE>"):
                        state = (
                            line.replace("<STATE>", "")
                            .replace("</STATE>", "")
                            .strip()
                            .upper()
                        )
                        with self.lock:
                            if state == "STOPPED":
                                if self.state != "STOPPED":
                                    logger.info("Arduino: STOPPED")
                                self.state = "STOPPED"
                                self.car.moving = False  # Stop car movement
                                self.car.state_reason = "Obstacle detected by Arduino"
                                self.car.arduino_obstacle_detected = (
                                    True  # Set the flag
                                )
                                # Send command to stop the servo
                                self.game.send_command("STOP_SERVO")
                            elif state == "MOVING":
                                if self.state != "MOVING":
                                    logger.info("Arduino: MOVING")
                                self.state = "MOVING"
                                self.car.arduino_obstacle_detected = (
                                    False  # Reset the flag
                                )
                                if self.car.current_target:
                                    self.car.moving = (
                                        True  # Resume movement if target is set
                                    )
                                    self.car.state_reason = "Moving towards waypoint"
                                    # Send command to start the servo
                                    self.game.send_command("START_SERVO")
        except serial.SerialException as e:
            logger.error(f"Serial Exception: {e}")
            self.running = False

    def send_command(self, command):
        """
        Send a command to the Arduino via serial.

        Args:
            command (str): The command string to send.
        """
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(f"{command}\n".encode("utf-8"))
                logger.info(f"Sent command to Arduino: {command}")
            except serial.SerialException as e:
                logger.error(f"Failed to send command '{command}': {e}")

    def stop(self):
        """Stop the serial reader thread and close the serial connection."""
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
            logger.info("Serial connection closed.")


# -------------------- Game Class ---------------------#


class Game:
    """Main game class handling the game loop and rendering."""

    def __init__(self):
        """Initialize the game, including screen, clock, walls, waypoints, and car."""
        self.screen = pygame.display.set_mode((WIDTH, TOTAL_HEIGHT))
        pygame.display.set_caption("Autonomous Car Navigation")
        self.clock = pygame.time.Clock()
        self.walls = self.create_walls()
        self.define_waypoints()
        self.car = CarRobot(
            self.waypoints[0][0],
            self.waypoints[0][1],
            self.waypoints,
            self.waypoint_names,
            self.walls,
            prompt_queue,  # Pass the prompt_queue to the car
        )

        # Set the waypoint dictionary in the car
        self.car.waypoint_dict = self.waypoint_dict

        # Initialize serial communication for obstacle detection
        self.serial_reader = SerialReader(SERIAL_PORT, BAUD_RATE, self.car, self)
        self.serial_reader.start()

        # State tracking for command sending
        self.previous_moving_state = False  # Initially not moving

    def create_walls(self):
        """
        Define and create wall objects in the environment.

        Returns:
            list: List of Wall objects.
        """
        wall_rects = [
            pygame.Rect(50, 50, 700, 50),  # Top wall
            pygame.Rect(50, 500, 700, 50),  # Bottom wall
            pygame.Rect(50, 50, 50, 500),  # Left wall
            pygame.Rect(720, 50, 50, 500),  # Right wall moved from x=700 to x=720
            pygame.Rect(350, 250, 100, 100),  # Middle wall
        ]
        return [Wall(rect) for rect in wall_rects]

    def define_waypoints(self):
        """
        Define the waypoints for the car to navigate.
        """
        self.waypoints = [
            (150, 150),  # Start
            (600, 150),  # M215 moved from x=650 to x=600
            (600, 450),  # M216 moved from x=650 to x=600
            (150, 450),  # M217
        ]
        self.waypoint_names = [
            "Start",
            "M215",
            "M216",
            "M217",
        ]

        # Map waypoint names to positions
        self.waypoint_dict = {
            name: position
            for name, position in zip(self.waypoint_names, self.waypoints)
        }

    def draw_walls(self):
        """Draw all walls on the screen."""
        for wall in self.walls:
            wall.draw(self.screen)

    def choose_waypoint(self, mouse_x, mouse_y):
        """Choose a waypoint based on mouse click and update the car's target."""
        closest_index = None
        min_distance = float("inf")

        for idx, (wp_x, wp_y) in enumerate(self.waypoints):
            distance = math.hypot(mouse_x - wp_x, mouse_y - wp_y)
            if (
                distance < min_distance and distance < 50
            ):  # Threshold for clicking accuracy
                closest_index = idx
                min_distance = distance

        if closest_index is not None:
            destination_name = self.waypoint_names[closest_index]
            target_point = self.waypoints[closest_index]
            self.car.set_target(target_point, destination_name)
            logger.info(
                f"Selected waypoint {destination_name}: ({target_point[0]}, {target_point[1]})"
            )
            # Send 'START_SERVO' command to Arduino
            self.send_command("START_SERVO")

    def send_command(self, command):
        """
        Send a command to the Arduino via serial.

        Args:
            command (str): The command string to send.
        """
        self.serial_reader.send_command(command)

    def process_commands(self):
        """Process any incoming commands from the command queue."""
        try:
            while True:
                command = command_queue.get_nowait()
                logger.info(f"Processing command from Flask: {command}")
                if command.startswith("go_to_"):
                    # Extract room and get target point
                    room = command.split("_")[2]
                    if room in self.waypoint_names:
                        target_point = self.waypoint_dict[room]
                        self.car.set_target(target_point, room)
                        logger.info(
                            f"Set target to {room}: ({target_point[0]}, {target_point[1]})"
                        )
                        # Send 'START_SERVO' command to Arduino
                        self.send_command("START_SERVO")
                    else:
                        logger.warning(f"Unknown room: {room}")
                elif command == "user_choice_done":
                    # User chose 'Done' - return to start
                    logger.info("User chose 'Done' - returning to start.")
                    self.car.return_to_start()
                    # Enqueue 'Goodbye' to response_queue
                    response_queue.put("Goodbye, going to start point.")
                elif command == "user_choice_go_another":
                    # User chose 'Go Another' - waiting for new waypoint
                    logger.info("User chose 'Go Another' - waiting for new waypoint.")
                    self.car.state_reason = "Waiting for waypoint"
                    # Enqueue 'Where do you want to go next' to response_queue
                    response_queue.put("Where do you want to go next")

        except queue.Empty:
            pass  # No commands to process

    def process_responses(self):
        """Process any incoming responses from the response queue."""
        try:
            while True:
                response = response_queue.get_nowait()
                logger.info(f"Processing response: {response}")
                # Handle TTS in a separate thread to avoid blocking
                threading.Thread(
                    target=self.perform_tts, args=(response,), daemon=True
                ).start()
        except queue.Empty:
            pass  # No responses to process

    def perform_tts(self, text):
        """Generate and play TTS audio for the given text."""
        try:
            temp_file = f"response_{uuid.uuid4()}.mp3"
            tts = gTTS(text=text, lang="en")
            tts.save(temp_file)
            logger.info(f"TTS audio saved as {temp_file}")

            # Play the generated speech audio
            sound = AudioSegment.from_mp3(temp_file)
            play(sound)

            # Remove the audio file after playback
            os.remove(temp_file)
            logger.info(f"TTS audio file {temp_file} removed after playback.")
        except Exception as e:
            logger.error(f"Error in perform_tts: {e}")

    def run_flask_app(self):
        """Run the Flask app."""
        # URL to open
        url = "http://127.0.0.1:5000/"

        # Start a thread to open the browser after a short delay
        threading.Thread(
            target=open_browser_after_delay, args=(url,), daemon=True
        ).start()

        # Running Flask server on port 5000
        app.run(debug=False, port=5000, use_reloader=False)

    def run(self):
        """Main game loop."""
        # Start Flask app in a separate thread
        flask_thread = threading.Thread(target=self.run_flask_app, daemon=True)
        flask_thread.start()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    if mouse_y < HEIGHT:
                        self.choose_waypoint(mouse_x, mouse_y)

            # Process any incoming commands from Flask
            self.process_commands()

            # Process any incoming responses from Flask
            self.process_responses()

            # Update car movement
            self.car.update()

            # State Tracking: Send commands based on state transitions
            current_moving_state = self.car.moving
            if current_moving_state != self.previous_moving_state:
                if current_moving_state:
                    # Car started moving
                    self.send_command("START_SERVO")
                    logger.debug(
                        "Command 'START_SERVO' sent due to state transition to MOVING."
                    )
                else:
                    # Car stopped moving
                    self.send_command("STOP_SERVO")
                    logger.debug(
                        "Command 'STOP_SERVO' sent due to state transition to STOPPED."
                    )
            self.previous_moving_state = current_moving_state

            # Render environment and car
            self.screen.fill(WHITE)
            self.draw_walls()
            self.car.draw(self.screen)
            self.car.draw_status(self.screen)  # Draw car status on the screen

            # Update the display and tick the clock
            pygame.display.flip()
            self.clock.tick(FPS)

        # Clean up and stop serial reader thread
        self.serial_reader.stop()
        self.serial_reader.join()

        pygame.quit()
        sys.exit()


# -------------------- Main Block ---------------------#


if __name__ == "__main__":
    try:
        game = Game()
        game.run()
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
