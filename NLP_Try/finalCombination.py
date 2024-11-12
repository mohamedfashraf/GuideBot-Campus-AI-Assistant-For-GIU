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
from threading import Lock
from datetime import datetime

# Initialize thread-safe queues
command_queue = queue.Queue()
response_queue = queue.Queue()
prompt_queue = queue.Queue()  # Queue for prompts

# -------------------- Constants ---------------------#

# Pygame Initialization
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
BUTTON_AREA_HEIGHT = 50  # Space for buttons and prompt
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
SENSOR_LENGTH = 45  # Adjusted sensor length
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

# Initialize the zero-shot classification pipeline with adjusted parameters
# Initialize the zero-shot classification pipeline with multi-label enabled
nlp = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    clean_up_tokenization_spaces=True,
    multi_label=True,  # Enable multi-label classification
)


VALID_ROOMS = {
    "M215",
    "M216",
    "Admission",
    "Financial",
    "Student_Affairs",
}  # Add more rooms as needed

# Define separate labels for intents and rooms
intents = [
    "Greeting",  # Handles "hi", "hey", "hello"
    "Farewell",  # Handles "bye", "goodbye", etc.
    "Apply",  # Intent to apply for something
    "Enroll",  # Intent to enroll in something
    "Navigate",  # Intent to navigate to a location
    "Payment",  # Intent to make a payment
    "Registration",  # Intent to register for something
    "Help",  # General help intent
]

rooms = list(VALID_ROOMS)  # Rooms defined earlier: "M215", "M216", etc.

# Combine intents and rooms into a single labels list
labels = intents + rooms

# Add dynamic command variations for each room in VALID_ROOMS
for room in VALID_ROOMS:
    # Add room-specific intents with synonyms and varied expressions
    labels.extend(
        [
            f"Go to {room}",
            f"Navigate to {room}",
            f"Take me to {room}",
            f"Go to the {room} office",
            f"Navigate to the {room} office",
            f"Take me to the {room} office",
            f"Apply at {room}",
            f"Apply for {room}",
            f"Apply for {room} office",
            f"Enroll at {room}",
            f"Registration at {room}",
            f"Payment at {room}",
            f"Course registration at {room}",
            f"Submit application at {room}",
            f"{room} application",
            f"Financial payment at {room}",
            f"Make a payment at {room}",
            f"Pay fees at {room}",
            f"Student affairs enrollment at {room}",
            f"Add course at {room}",
            f"Drop course at {room}",
            f"Financial transaction at {room}",
            f"Apply for the next semester at {room}",  # Specific user input
        ]
    )

weekly_schedule = {
    "Financial": {
        "saturday": {"opens_at": "09:00", "closes_at": "17:00"},
        "sunday": {"opens_at": "09:00", "closes_at": "17:00"},
        "monday": {"opens_at": "09:00", "closes_at": "17:00"},
        "tuesday": {"opens_at": "09:00", "closes_at": "17:00"},
        "wednesday": {"opens_at": "09:00", "closes_at": "17:00"},
        "thursday": {"opens_at": "09:00", "closes_at": "17:00"},
    },
    "Student_Affairs": {
        "saturday": {"opens_at": "10:00", "closes_at": "16:00"},
        "sunday": {"opens_at": "10:00", "closes_at": "16:00"},
        "monday": {"opens_at": "10:00", "closes_at": "16:00"},
        "tuesday": {"opens_at": "10:00", "closes_at": "16:00"},
        "wednesday": {"opens_at": "10:00", "closes_at": "16:00"},
        "thursday": {"opens_at": "10:00", "closes_at": "16:00"},
    },
    "Admission": {
        "saturday": {"opens_at": "08:00", "closes_at": "15:00"},
        "sunday": {"opens_at": "08:00", "closes_at": "15:00"},
        "monday": {"opens_at": "08:00", "closes_at": "15:00"},
        "tuesday": {"opens_at": "08:00", "closes_at": "24:00"},
        "wednesday": {"opens_at": "08:00", "closes_at": "15:00"},
        "thursday": {"opens_at": "08:00", "closes_at": "15:00"},
    },
}

def check_room_availability(room):
    """Function to check room availability based on the current day and time."""
    current_day = datetime.now().strftime("%A").lower()
    current_time = datetime.now().strftime("%H:%M")

    if room in weekly_schedule and current_day in weekly_schedule[room]:
        opening_time = weekly_schedule[room][current_day]["opens_at"]
        closing_time = weekly_schedule[room][current_day]["closes_at"]

        # Check if the current time is within the room's open hours
        if opening_time <= current_time <= closing_time:
            return {
                "is_open": True,
                "opens_at": opening_time,
                "closes_at": closing_time,
            }
        else:
            return {
                "is_open": False,
                "opens_at": opening_time,
                "closes_at": closing_time,
            }

    # Default response if no schedule found
    return {"is_open": True}


app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  # Enable CORS for all routes if frontend is on a different origin

# State management for conversation
pending_action = None
pending_action_lock = Lock()

import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)


def open_application(command_label):
    global pending_action
    response = ""

    logger.debug(f"open_application called with command_label: {command_label}")

    # Step 1: Handle Greeting and Farewell Intents
    if command_label == "Greeting":
        response = "Hello there! How can I assist you today?"
        response_queue.put(response)
        logger.info(f"Responding: {response}")
        return jsonify({"response": response})

    elif command_label == "Farewell":
        response = "Goodbye! Have a great day."
        response_queue.put(response)
        logger.info(f"Responding: {response}")
        return jsonify({"response": response})

    # Step 2: Handle Intent and Room Commands
    elif " at " in command_label:
        # Split the command_label into intent and room
        intent, room = command_label.split(" at ", 1)
        intent = intent.strip()
        room = room.strip()

        logger.debug(f"Handling intent '{intent}' for room '{room}'")

        if room not in VALID_ROOMS:
            response = f"I couldn't find the {room} office. Could you please specify a valid room?"
            response_queue.put(response)
            return jsonify({"response": response})

        if intent in ["Apply", "Enroll", "Navigate", "Payment", "Registration"]:
            if intent == "Navigate":
                action = "navigate"
            else:
                action = intent.lower()

            # Provide initial conversational response based on identified room
            response = f"It sounds like you want to {action} at the {room} office. Would you like me to check if it's open?"
            with pending_action_lock:
                pending_action = f"check_availability_{room}"
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})
        else:
            response = "I'm not sure how to assist with that action. Could you please provide more details?"
            response_queue.put(response)
            return jsonify({"response": response})

    # Step 3: Check availability if pending action indicates it and user confirms
    elif (
        command_label == "CONFIRM_YES"
        and pending_action
        and pending_action.startswith("check_availability_")
    ):
        room = pending_action.split("_")[-1]
        logger.debug(f"Checking availability for room: {room}")
        availability = check_room_availability(room)

        if availability["is_open"]:
            response = f"{room} is open. Would you like me to guide you there?"
            with pending_action_lock:
                pending_action = f"go_to_{room}"
            logger.info(f"Responding: {response}")
        else:
            next_open_day, next_open_time = get_next_opening(room)
            response = f"{room} is currently closed and will open on {next_open_day} at {next_open_time}. Would you like help with something else?"

            # Set pending action to handle further assistance request
            with pending_action_lock:
                pending_action = "ask_if_help_needed"
            logger.info(f"Responding: {response}")

        response_queue.put(response)
        return jsonify({"response": response})

    # Step 4: Handle further assistance request based on user response
    elif pending_action == "ask_if_help_needed":
        if command_label == "YES":
            response = "Great! What would you like help with?"
            pending_action = None  # Reset pending_action to allow new commands
        elif command_label == "NO":
            response = "Fine, do you need any other help?"
            with pending_action_lock:
                pending_action = "handle_additional_help"
        else:
            # Fast forward if user states their request directly
            pending_action = None
            logger.debug("User provided a direct request instead of YES/NO.")
            return open_application(
                command_label
            )  # Re-run to handle the direct request

        response_queue.put(response)
        logger.info(f"Responding: {response}")
        return jsonify({"response": response})

    # Step 6: Handle the response to "Fine, do you need any other help?"
    elif pending_action == "handle_additional_help":
        if command_label == "YES":
            response = "Please provide your next command."
            pending_action = None  # Allow the user to input a new command
        elif command_label == "NO":
            response = "Okay, returning to the start point."
            # Queue the command to navigate to the Start point
            command_queue.put("go_to_Start")
            pending_action = None  # Reset pending_action
        else:
            # If the user provides an unrecognized response
            response = "I didn't understand that. Returning to the start point."
            command_queue.put("go_to_Start")
            pending_action = None  # Reset pending_action

        response_queue.put(response)
        logger.info(f"Responding: {response}")
        return jsonify({"response": response})

    # Step 5: Guide the user to the room if they confirm guidance
    elif (
        command_label == "CONFIRM_YES"
        and pending_action
        and pending_action.startswith("go_to_")
    ):
        room = pending_action.split("_")[-1]
        logger.debug(f"Command received: guiding to {room}")  # Dynamic room reference
        command_queue.put(f"go_to_{room}")
        response = f"Taking you to the {room} room now."
        pending_action = None
        response_queue.put(response)
        logger.info(f"Responding: {response}")
        return jsonify({"response": response})

    elif command_label == "CONFIRM_NO" and pending_action:
        response = "Okay, let me know if you need anything else."
        pending_action = None
        response_queue.put(response)
        logger.info(f"Responding: {response}")
        return jsonify({"response": response})

    # Handle greetings or fallback for unrecognized commands
    elif command_label in ["Greeting", "hi", "hey", "hello"]:
        response = "Hello there! How can I assist you today?"
    elif command_label in ["Farewell", "bye", "goodbye"]:
        response = "Goodbye! Have a great day."
    else:
        response = "I'm not sure what you meant. Could you please rephrase?"

    response_queue.put(response)
    logger.info(f"Responding: {response}")
    return jsonify({"response": response})


# Helper function to find the next opening day and time
def get_next_opening(room):
    current_day = datetime.now().strftime("%A").lower()
    current_time = datetime.now().strftime("%H:%M")
    days_of_week = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]

    # Find the next day and time the room is open
    for i in range(7):  # Check up to a week ahead
        day_index = (days_of_week.index(current_day) + i) % 7
        next_day = days_of_week[day_index]
        if next_day in weekly_schedule[room]:
            opening_time = weekly_schedule[room][next_day]["opens_at"]
            if i > 0 or current_time < opening_time:
                return next_day.capitalize(), opening_time

    return (
        None,
        None,
    )  # If no opening time is found, which is unlikely in a weekly schedule


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/command", methods=["POST"])
def handle_command():
    """Handles the POST request from the frontend, processes the command, and sends back a response."""
    data = request.json
    logger.debug(f"Received data: {data}")
    command_text = data.get("text", "").strip()

    if command_text:
        logger.debug(f"Command received: {command_text}")

        # Define a comprehensive hypothesis template
        hypothesis_template = "The user wants to {}."

        result = nlp(
            command_text,
            candidate_labels=labels,
            hypothesis_template=hypothesis_template,
            multi_label=True,  # Enable multi-label predictions
        )

        # Log the model's output to understand prediction
        logger.debug(f"Model result: {result}")
        predicted_labels = result["labels"]
        scores = result["scores"]

        logger.info(f"Predicted labels: {predicted_labels} with scores {scores}")

        # Set a confidence threshold
        threshold = 0.2  # Adjust based on testing

        # Extract intents and rooms based on the confidence threshold
        detected_intents = [
            label
            for label, score in zip(predicted_labels, scores)
            if label in intents and score > threshold
        ]
        detected_rooms = [
            label
            for label, score in zip(predicted_labels, scores)
            if label in rooms and score > threshold
        ]

        logger.info(f"Detected intents: {detected_intents}")
        logger.info(f"Detected rooms: {detected_rooms}")

        response = ""

        if detected_intents:
            intent = detected_intents[0]  # Assume the highest confidence intent
            logger.debug(f"Handling intent: {intent}")

            if intent in ["Greeting", "Farewell"]:
                return open_application(intent)
            elif intent in [
                "Apply",
                "Enroll",
                "Navigate",
                "Payment",
                "Registration",
                "Help",
            ]:
                if detected_rooms:
                    room = detected_rooms[0]  # Assume the highest confidence room
                    return open_application(f"{intent} at {room}")
                else:
                    # Handle cases where intent is detected but room isn't
                    response = f"It sounds like you want to {intent.lower()}. Which office or room are you referring to?"
                    response_queue.put(response)
                    return jsonify({"response": response})
            else:
                # Handle other intents if necessary
                response = "I'm not sure how to assist with that. Could you please provide more details?"
                response_queue.put(response)
                return jsonify({"response": response})
        else:
            response = "I'm not sure what you meant. Could you please rephrase?"
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
        self.obstacle_response_sent = False  # Initialize flag to prevent early responses
        self.started_moving = False

        # Initialize waypoint_dict inside __init__
        self.waypoint_dict = {
            name: position for name, position in zip(self.waypoint_names, self.waypoints)
        }
        logger.info(f"Waypoint Dictionary Initialized: {self.waypoint_dict}")

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

    def rotate(self, angle_change):
        """
        Rotate the car by a certain angle change, with collision checking.

        Args:
            angle_change (float): The angle change in degrees.
        """
        original_angle = self.angle  # Save original angle
        self.angle = (self.angle + angle_change) % 360
        self.update_sensors()

        # Check for collision after rotation
        if self.check_collision(self.x, self.y):
            # If collision occurs, revert to original angle
            self.angle = original_angle
            self.update_sensors()
            self.state_reason = "Cannot rotate due to collision"
            self.moving = False  # Stop moving

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
            angle_change = angle_diff
        elif angle_diff > 0:
            angle_change = CAR_ROTATION_SPEED
        else:
            angle_change = -CAR_ROTATION_SPEED

        self.rotate(angle_change)

    def move_forward(self):
        """Move the car forward based on its speed and angle."""
        new_x = self.x + self.speed * math.cos(math.radians(self.angle))
        new_y = self.y - self.speed * math.sin(math.radians(self.angle))
        if not self.check_collision(new_x, new_y):
            self.x = new_x
            self.y = new_y
            self.update_sensors()
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
            self.x, self.y = (
                original_x,
                original_y,
            )  # Reset position if collision occurs
        return collision


    def set_target(self, target_point, destination_name):
        self.destination_name = destination_name
        self.moving = True
        self.state_reason = f"Moving towards {destination_name}"
        self.update_sensors()
        path_key = (self.current_location_name, destination_name)

        if path_key in self.waypoint_paths:
            self.path = [
                self.waypoint_dict[wp_name] for wp_name in self.waypoint_paths[path_key]
            ]
            self.current_target = self.path.pop(0)
            logger.debug(f"Path found for {path_key}: {self.path}")
        else:
            self.current_target = target_point
            self.path = []
            logger.debug(f"No predefined path for {path_key}. Directly setting target.")

        logger.info(f"Set target for {destination_name}: {self.current_target}")

    def update(self):
        """Update the car's state by rotating and moving towards the target."""
        if self.moving and self.current_target:
            # Set started_moving to True once movement begins
            self.started_moving = True

            # First, check if Arduino has detected an obstacle
            if self.arduino_obstacle_detected:
                self.moving = False
                self.state_reason = "Obstacle detected by Arduino"
                # Send polite voice response only if not already sent and movement has started
                if self.started_moving and not self.obstacle_response_sent:
                    response_queue.put("Excuse me, could you please let me pass?")
                    self.obstacle_response_sent = True
                return

            # Check sensors for obstacles
            sensor_data = self.check_sensors()
            obstacles = [detected for angle, detected in sensor_data if detected]

            # If close to the target, disable obstacle avoidance
            target_distance = math.hypot(
                self.current_target[0] - self.x, self.current_target[1] - self.y
            )
            if target_distance < self.threshold * 2:
                obstacles = []

            if obstacles:
                # Obstacle detected by sensors, stop and send response
                self.moving = False
                self.state_reason = "Waiting for obstacle to clear"
                if self.started_moving and not self.obstacle_response_sent:
                    response_queue.put(
                        "Hi! I’m the campus GuideBot. Could you help clear the way?"
                    )
                    self.obstacle_response_sent = True
            else:
                # Clear the flag if no obstacles are detected and proceed towards target
                self.obstacle_response_sent = False
                target_angle = self.get_target_angle()
                angle_diff = (target_angle - self.angle + 360) % 360
                if angle_diff > 180:
                    angle_diff -= 360

                if abs(angle_diff) > CAR_ROTATION_SPEED:
                    # Rotate towards target
                    self.rotate_towards_target(target_angle)
                    self.state_reason = "Rotating towards target"
                else:
                    # Move forward
                    self.move_forward()
                    self.check_point_reached()
        else:
            # If not moving, check for Arduino obstacle response only if movement has started
            if self.arduino_obstacle_detected:
                self.state_reason = "Obstacle detected by Arduino"
                if self.started_moving and not self.obstacle_response_sent:
                    response_queue.put("Excuse me, could you please let me pass?")
                    self.obstacle_response_sent = True
            else:
                self.state_reason = "Stopped"
                self.obstacle_response_sent = False  # Reset the flag when stopped

    def draw_status(self, surface):
        font = pygame.font.SysFont(None, 24)
        status = "MOVING" if self.moving else "STOPPED"

        # Render status and reason as text
        status_text = font.render(f"Robot Status: {status}", True, BLACK)
        reason_text = font.render(f"Reason: {self.state_reason}", True, BLACK)

        # Determine the width of the status text to place reason text with a gap
        status_width = status_text.get_width()
        margin = 20  # Adjust this value for more or less space between texts

        # Draw status and reason side by side with a margin
        surface.blit(status_text, (10, HEIGHT + 10))
        surface.blit(reason_text, (10 + status_width + margin, HEIGHT + 10))

    def draw(self, surface):
        """Render the car and waypoints on the screen."""
        rotated_image, rect = self.update_mask()
        surface.blit(rotated_image, rect.topleft)

        # Draw sensors with color indicating obstacle detection
        sensor_data = self.check_sensors()
        for (sensor_angle, (sensor_end_x, sensor_end_y)), (
            _,
            obstacle_detected,
        ) in zip(self.sensors, sensor_data):
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

    # Define the predefined paths between waypoints with consistent casing
    waypoint_paths = {
        ("Start", "M215"): ["M215"],
        ("M215", "M216"): ["M216"],
        ("M216", "Admission"): ["Admission"],
        ("Admission", "Start"): ["M216", "M215", "Start"],  # Updated path to avoid blocked route
        ("Start", "M216"): ["M215", "M216"],
        ("Start", "Admission"): ["M215", "M216", "Admission"],  # Must go through M215 and M216
        ("M215", "Admission"): ["M216", "Admission"],
        ("M216", "M215"): ["M215"],
        ("M216", "Start"): ["M215", "Start"],
        ("Admission", "M216"): ["M216"],
        ("Admission", "M215"): ["M216", "M215"],
        ("M215", "Start"): ["Start"],
        # Add more paths as needed
    }

    # Map waypoint names to positions with consistent casing
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
        self.game = game  # Reference to the Game for sending commands
        self.state = "STOPPED"  # Initialize to STOPPED
        self.lock = threading.Lock()
        self.obstacle_response_sent = False  # Add this line

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
        pygame.display.set_caption("GuideBot Simulation")
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
            pygame.Rect(720, 50, 50, 500),  # Right wall
            pygame.Rect(350, 250, 100, 100),  # Middle wall
            pygame.Rect(50, 300, 300, 10),  # Wall to separate Start and Admission
        ]
        return [Wall(rect) for rect in wall_rects]

    def define_waypoints(self):
        """
        Define the waypoints for the car to navigate.
        """
        self.waypoints = [
            (150, 150),  # Start
            (600, 150),  # M215
            (600, 450),  # M216
            (150, 450),  # Admission
        ]
        self.waypoint_names = [
            "Start",
            "M215",
            "M216",
            "Admission",
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
        try:
            while True:
                command = command_queue.get_nowait()
                logger.info(f"Processing command from Flask: {command}")
                if command.startswith("go_to_"):
                    # Extract the room name dynamically, accounting for possible underscores
                    parts = command.split("_", 2)
                    if len(parts) >= 3:
                        room = parts[2]
                    else:
                        room = parts[-1]

                    # Handle rooms with underscores (e.g., "Student_Affairs")
                    room = room.replace("_", " ")  # Optional: Adjust based on naming

                    # Check if the room exists in waypoint_names
                    if room in self.waypoint_names:
                        target_point = self.waypoint_dict[room]
                        self.car.set_target(target_point, room)
                        logger.info(f"Setting target to {room}: {target_point}")
                        self.send_command("START_SERVO")
                    else:
                        logger.warning(f"Unknown room: {room}")
                elif command == "user_choice_done":
                    self.car.return_to_start()
                    response_queue.put("Goodbye, going to start point.")
                elif command == "user_choice_go_another":
                    self.car.state_reason = "Waiting for waypoint"
                    response_queue.put("Where do you want to go next")
        except queue.Empty:
            pass

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
