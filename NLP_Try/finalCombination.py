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
from transformers import pipeline
from pydub import AudioSegment
from pydub.playback import play
import uuid
import webbrowser
from threading import Lock
from datetime import datetime
import re  # Import regex module for pattern matching
from functools import lru_cache  # Import lru_cache for caching
from concurrent.futures import (
    ThreadPoolExecutor,
)  # Import ThreadPoolExecutor for asynchronous processing

# Initialize thread-safe queues for inter-thread communication
command_queue = queue.Queue()
response_queue = queue.Queue()
prompt_queue = queue.Queue()  # Queue for prompts to be sent to the frontend

# -------------------- Constants ---------------------#

# Initialize Pygame
pygame.init()

# Define screen dimensions
WIDTH, HEIGHT = 800, 600
BUTTON_AREA_HEIGHT = 50  # Space reserved for buttons and prompts
TOTAL_HEIGHT = HEIGHT + BUTTON_AREA_HEIGHT  # Total screen height including button area

# Define color constants using RGB tuples
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Car properties
CAR_IMAGE_PATH = "navigationTry/2d-super-car-top-view.png"  # Path to the car image
CAR_SIZE = (80, 40)  # Size of the car image (width, height)
CAR_SPEED = 2  # Speed at which the car moves
CAR_ROTATION_SPEED = 5  # Degrees the car rotates per update

# Sensor properties for obstacle detection
SENSOR_LENGTH = 45  # Length of each sensor
SENSOR_ANGLES = [-30, 0, 30]  # Angles for left, front, and right sensors

# Waypoint properties
WAYPOINT_THRESHOLD = 20  # Distance threshold to consider a waypoint as reached

# Frame rate for the Pygame loop
FPS = 60

# Serial communication settings for Arduino
SERIAL_PORT = "COM5"  # Serial port (update as per your system)
BAUD_RATE = 115200  # Baud rate matching Arduino's settings

# -------------------- Logging Configuration ---------------------#

# Configure the logging module to display DEBUG level logs and above
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log output to the console
    ],
)

# Create a logger object for this module
logger = logging.getLogger(__name__)

# -------------------- Flask App Setup ---------------------#

# Initialize the zero-shot classification pipeline using a pre-trained model
nlp = pipeline(
    "zero-shot-classification",
    model="microsoft/deberta-base-mnli",  # Switched to a more efficient model
    tokenizer="microsoft/deberta-base-mnli",
    framework="pt",  # Use PyTorch framework
    device=-1,  # Use CPU (set to a GPU device index if available)
)

# Define valid rooms with consistent naming (Title Case, underscores)
VALID_ROOMS = {
    "M215",
    "M216",
    "Admission",
    "Financial",
    "Student_Affairs",
}

# Define valid doctors with underscores
VALID_DOCTORS = {
    "dr_nada",
    "dr_slim",
    "dr_omar",
}

# List of days of the week for schedule queries
DAYS_OF_WEEK = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]

# Expanded labels with synonyms and related terms for NLP classification
labels = [
    "kill",
    "ask_admission_open",
    "confirm_yes",
    "confirm_no",
    "none",
    "hi",
    "hey",
    "hello",
    # Financial matters
    "financial",
    "finance office",
    "tuition payment",
    "pay tuition",
    "scholarship",
    "billing",
    # Student affairs
    "student affairs",
    "course enrollment",
    "add course",
    "drop course",
    "class schedule",
    "enrollment services",
    # Admissions
    "admission",
    "apply to university",
    "application",
    "enroll",
    "registration",
    "apply",
    # Computer Science Major
    "computer science major",
    "cs major",
    "tell me about computer science",
    "i want to study computer science",
    "computer science information",
    "computer science department",
    "cs department",
    # Doctors
    "dr_nada",
    "see dr_nada",
    "visit dr_nada",
    "dr_slim",
    "see dr_slim",
    "visit dr_slim",
    "dr_omar",
    "see dr_omar",
    "visit dr_omar",
]

# Dynamically add command variations for each room
for room in VALID_ROOMS:
    room_lower = room.replace("_", " ").lower()
    labels.append(room_lower)
    labels.extend(
        [
            f"go to {room_lower}",
            f"navigate to {room_lower}",
            f"take me to {room_lower}",
            f"go to room {room_lower}",
            f"take me to room {room_lower}",
            f"is {room_lower} open",
            f"what are the opening times for {room_lower}",
            f"when does {room_lower} open",
            f"when does {room_lower} close",
            f"what time does {room_lower} open",
            f"what time does {room_lower} close",
        ]
    )

# Dynamically add command variations for each doctor
for doctor in VALID_DOCTORS:
    doctor_lower = doctor.replace("_", " ").lower()
    labels.append(doctor_lower)
    labels.extend(
        [
            f"is {doctor_lower} available",
            f"when is {doctor_lower} available",
            f"what are {doctor_lower}'s working hours",
            f"what time is {doctor_lower} available",
            f"can I meet {doctor_lower}",
            f"see {doctor_lower}",
            f"visit {doctor_lower}",
            f"go to {doctor_lower}",
        ]
    )

# Add days of the week and "now" to labels for temporal queries
labels.extend(DAYS_OF_WEEK)
labels.append("now")

# Additional labels for GIU-specific interactions
labels.extend(
    [
        "giu",
        "german international university",
        "apply to giu",
        "admission at giu",
        "next semester at giu",
        "apply for next semester",
    ]
)

# Define weekly schedule for different rooms with opening and closing times
weekly_schedule = {
    "Financial": {
        day: {"opens_at": "09:00", "closes_at": "17:00"} for day in DAYS_OF_WEEK
    },
    "Student_Affairs": {
        day: {"opens_at": "10:00", "closes_at": "18:00"} for day in DAYS_OF_WEEK
    },
    "Admission": {
        day: {"opens_at": "08:00", "closes_at": "16:00"} for day in DAYS_OF_WEEK
    },
}


def create_doctor_schedule():
    """Helper function to create a schedule for each doctor."""
    schedule = {}
    common_times = [
        "08:30 - 10:00",
        "10:15 - 11:45",
        "12:00 - 13:30",
        "15:45 - 17:15",
    ]
    for doctor in VALID_DOCTORS:
        schedule[doctor] = {day: common_times.copy() for day in DAYS_OF_WEEK}
    return schedule


# Initialize doctor availability schedules
doctor_availability = create_doctor_schedule()


def check_room_availability(room):
    """
    Function to check room availability based on the current day and time.

    Args:
        room (str): The name of the room to check.

    Returns:
        dict: Dictionary indicating if the room is open, and its opening and closing times.
    """
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

    # Default response if no schedule is found
    return {"is_open": True}


# Initialize Flask application with CORS enabled
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  # Enable CORS for all routes if frontend is on a different origin

# State management variable for conversation flow
pending_action = None
pending_action_lock = Lock()

# Initialize ThreadPoolExecutor for asynchronous NLP processing
executor = ThreadPoolExecutor(max_workers=4)  # Adjust based on CPU cores


@lru_cache(maxsize=128)
def classify_command_cached(command_text):
    """
    Classify the command using zero-shot classification with caching.

    Args:
        command_text (str): The user's input command.

    Returns:
        str: The highest confidence label or 'none' if no label meets the threshold.
    """
    result = nlp(
        command_text,
        candidate_labels=tuple(labels),  # Convert to tuple for hashing
        hypothesis_template="This text is about {}.",
        multi_label=True,
    )
    confidence_threshold = 0.3  # Adjust as needed
    matched_labels = [
        label
        for label, score in zip(result["labels"], result["scores"])
        if score > confidence_threshold
    ]
    return matched_labels[0] if matched_labels else "none"


def is_affirmative(response):
    """
    Check if the response is affirmative.

    Args:
        response (str): The user's response text.

    Returns:
        bool: True if affirmative, False otherwise.
    """
    affirmative_responses = [
        "yes",
        "yeah",
        "yep",
        "sure",
        "please do",
        "of course",
        "affirmative",
    ]
    return any(word in response.lower() for word in affirmative_responses)


def is_negative(response):
    """
    Check if the response is negative.

    Args:
        response (str): The user's response text.

    Returns:
        bool: True if negative, False otherwise.
    """
    negative_responses = ["no", "nope", "not now", "maybe later", "negative"]
    return any(word in response.lower() for word in negative_responses)


def open_application(command, original_command_text):
    """
    Process the command and handle various pending actions.

    Args:
        command (str): The normalized command label.
        original_command_text (str): The original user input text.

    Returns:
        Flask Response: JSON response to be sent back to the frontend.
    """
    global pending_action
    response = ""
    command = command.strip().lower()

    logger.debug(f"open_application called with command: {command}")
    logger.debug(f"Original command text: {original_command_text}")

    # Handle pending actions that require a simple yes/no response
    if pending_action and pending_action.startswith("go_to_"):
        if is_affirmative(original_command_text):
            location = pending_action[len("go_to_") :]
            location_normalized = location.replace("-", "_").replace(" ", "_").title()
            logger.debug(f"User confirmed to go to location: {location_normalized}")
            command_queue.put(f"go_to_{location_normalized}")
            if location_normalized.lower() in VALID_DOCTORS:
                response = f"Taking you to {location_normalized.replace('_', ' ')}'s office now."
            else:
                response = (
                    f"Taking you to the {location_normalized.replace('_', ' ')} now."
                )
            pending_action = None
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})

        elif is_negative(original_command_text):
            response = "Okay, let me know if you need anything else."
            pending_action = "ask_if_help_needed"
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})

        else:
            response = "I'm sorry, I didn't catch that. Please say yes or no."
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})

    elif pending_action == "ask_if_help_needed":
        if is_affirmative(original_command_text):
            response = "Great! What would you like help with?"
            pending_action = None
        elif is_negative(original_command_text):
            response = "Okay, feel free to ask if you need any assistance. Goodbye!"
            pending_action = None
        else:
            # Fast forward if user states their request directly
            pending_action = None
            logger.debug("User provided a direct request instead of YES/NO.")
            return open_application(
                command, original_command_text
            )  # Re-run to handle the direct request

        response_queue.put(response)
        logger.info(f"Responding: {response}")
        return jsonify({"response": response})

    # Handle pending action for checking doctor's availability
    elif pending_action and pending_action.startswith("check_doctor_availability_"):
        if is_affirmative(original_command_text):
            doctor_id = pending_action[len("check_doctor_availability_") :].replace(
                "-", "_"
            )
            availability = get_doctor_availability_data(doctor_id)
            if availability["is_available"]:
                response = f"{doctor_id.replace('_', ' ').title()} is available now. Would you like me to guide you to their office?"
                with pending_action_lock:
                    pending_action = f"go_to_{doctor_id}"
                logger.info(f"Responding: {response}")
            else:
                response = f"{doctor_id.replace('_', ' ').title()} is not available now. {availability['next_availability']} Would you like help with something else?"
                with pending_action_lock:
                    pending_action = "ask_if_help_needed"
                logger.info(f"Responding: {response}")

            response_queue.put(response)
            return jsonify({"response": response})

        elif is_negative(original_command_text):
            response = "Okay, let me know if you need anything else."
            pending_action = "ask_if_help_needed"
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})

        else:
            response = "I'm sorry, I didn't catch that. Please say yes or no."
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})

    # Handle pending action for asking day of week for room
    elif pending_action and pending_action.startswith("ask_for_day_room_"):
        room = pending_action[len("ask_for_day_room_") :]
        if is_negative(original_command_text):
            response = "Okay, let me know if you need anything else."
            pending_action = "ask_if_help_needed"
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})

        day_in_text = extract_day_from_text(original_command_text)
        if day_in_text:
            opening_times = get_room_opening_times(room, day_in_text)
            if opening_times:
                response = f"The {room.replace('_', ' ')} opens at {opening_times['opens_at']} and closes at {opening_times['closes_at']} on {day_in_text.capitalize()}."
            else:
                response = f"The {room.replace('_', ' ')} is closed on {day_in_text.capitalize()}."
            # Ask if the user needs anything else
            response += " Is there anything else I can assist you with?"
            pending_action = "ask_if_help_needed"
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})
        else:
            response = (
                "I'm sorry, I didn't catch the day. Please specify a day of the week."
            )
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})

    # Handle pending action for asking day of week for doctor
    elif pending_action and pending_action.startswith("ask_for_day_doctor_"):
        doctor = pending_action[len("ask_for_day_doctor_") :]
        if is_negative(original_command_text):
            response = "Okay, let me know if you need anything else."
            pending_action = "ask_if_help_needed"
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})

        day_in_text = extract_day_from_text(original_command_text)
        if day_in_text:
            schedule = get_doctor_schedule(doctor, day_in_text)
            if schedule:
                schedule_str = ", ".join(schedule)
                response = f"{doctor.replace('_', ' ').title()} is available at the following times on {day_in_text.capitalize()}: {schedule_str}."
            else:
                response = f"{doctor.replace('_', ' ').title()} is not available on {day_in_text.capitalize()}."
            # Ask if the user needs anything else
            response += " Is there anything else I can assist you with?"
            pending_action = "ask_if_help_needed"
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})
        else:
            response = (
                "I'm sorry, I didn't catch the day. Please specify a day of the week."
            )
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})

    # If no pending action requiring yes/no, proceed with normal processing
    # Normalize the command for matching
    command_normalized = command.replace("dr ", "").replace("doctor ", "").strip()

    # Step 1: Identify room or doctor availability queries using regex
    availability_query_match = re.search(
        r"(is|are|when|what time)(.*?)(open|close|available)", command_normalized
    )

    if availability_query_match:
        subject = availability_query_match.group(2).strip()
        logger.debug(f"Subject extracted for availability query: {subject}")

        # Normalize subject by removing titles and extra spaces
        subject_normalized = subject.replace("dr ", "").replace("doctor ", "").strip()

        # Extract day if specified in the command
        day_in_text = extract_day_from_text(command_normalized)
        logger.debug(f"Day extracted from query: {day_in_text}")

        # Check for 'now' in the command to represent the current day
        if "now" in command_normalized:
            day_in_text = datetime.now().strftime("%A").lower()
            logger.debug(
                f"'Now' detected, setting day_in_text to current day: {day_in_text}"
            )

        # Check if the subject is a room
        for room in VALID_ROOMS:
            room_lower = room.replace("_", " ").lower()
            if room_lower == subject_normalized or subject_normalized.endswith(
                room_lower
            ):
                if day_in_text:
                    opening_times = get_room_opening_times(room, day_in_text)
                    if opening_times:
                        response = f"The {room.replace('_', ' ')} opens at {opening_times['opens_at']} and closes at {opening_times['closes_at']} on {day_in_text.capitalize()}."
                    else:
                        response = f"The {room.replace('_', ' ')} is closed on {day_in_text.capitalize()}."
                    # Ask if the user needs anything else
                    response += " Is there anything else I can assist you with?"
                    pending_action = "ask_if_help_needed"
                    response_queue.put(response)
                    logger.info(f"Responding: {response}")
                    return jsonify({"response": response})
                else:
                    response = "Do you need the opening times for a specific day?"
                    with pending_action_lock:
                        pending_action = f"ask_for_day_room_{room}"
                    response_queue.put(response)
                    logger.info(f"Responding: {response}")
                    return jsonify({"response": response})

        # Check if the subject is a doctor
        for doctor in VALID_DOCTORS:
            doctor_lower = doctor.replace("_", " ").lower()
            doctor_normalized = (
                doctor_lower.replace("dr ", "").replace("doctor ", "").strip()
            )
            if (
                doctor_normalized == subject_normalized
                or doctor_normalized in subject_normalized
            ):
                if day_in_text:
                    schedule = get_doctor_schedule(doctor, day_in_text)
                    if schedule:
                        schedule_str = ", ".join(schedule)
                        response = f"{doctor.replace('_', ' ').title()} is available at the following times on {day_in_text.capitalize()}: {schedule_str}."
                    else:
                        response = f"{doctor.replace('_', ' ').title()} is not available on {day_in_text.capitalize()}."
                    response += " Is there anything else I can assist you with?"
                    pending_action = "ask_if_help_needed"
                    response_queue.put(response)
                    logger.info(f"Responding: {response}")
                    return jsonify({"response": response})
                else:
                    # Handle 'now' or no day specified
                    if not day_in_text:
                        day_in_text = datetime.now().strftime("%A").lower()
                    schedule = get_doctor_schedule(doctor, day_in_text)
                    if schedule:
                        schedule_str = ", ".join(schedule)
                        response = f"{doctor.replace('_', ' ').title()} is available at the following times today: {schedule_str}."
                    else:
                        response = f"{doctor.replace('_', ' ').title()} is not available today."
                    response += " Is there anything else I can assist you with?"
                    pending_action = "ask_if_help_needed"
                    response_queue.put(response)
                    logger.info(f"Responding: {response}")
                    return jsonify({"response": response})

    # Continue with previous logic if no availability queries detected

    # Step 1: Identify room or topic based on predefined rooms and keywords
    room = None
    doctor = None

    # Check if any valid room is mentioned in the command
    for valid_room in VALID_ROOMS:
        if valid_room.replace("_", " ").lower() in command:
            room = valid_room  # e.g., "Admission"
            logger.debug(f"Identified room: {room}")
            break

    # If no room identified, check for a doctor
    if not room:
        for valid_doctor in VALID_DOCTORS:
            doctor_lower = valid_doctor.replace("_", " ").lower()
            doctor_normalized = (
                doctor_lower.replace("dr ", "").replace("doctor ", "").strip()
            )
            if doctor_normalized in command_normalized:
                doctor = valid_doctor
                logger.debug(f"Identified doctor: {doctor}")
                break

    # Additional keywords for topics if no room or doctor is detected
    if not room and not doctor:
        if any(
            keyword in command
            for keyword in [
                "financial",
                "money",
                "payment",
                "pay",
                "tuition",
                "fee",
                "scholarship",
                "billing",
            ]
        ):
            room = "Financial"
        elif any(
            keyword in command
            for keyword in [
                "student affairs",
                "course",
                "enrollment",
                "add",
                "drop",
                "class schedule",
                "enrollment services",
            ]
        ):
            room = "Student_Affairs"
        elif any(
            keyword in command
            for keyword in [
                "admission",
                "apply",
                "application",
                "enroll",
                "registration",
                "apply to university",
                "apply for next semester",
                "next semester at giu",
                "admission at giu",
                "apply to giu",
            ]
        ):
            # Enhanced response for admission inquiries
            availability = check_room_availability("Admission")
            if availability["is_open"]:
                response = (
                    "We are thrilled that you're interested in joining the GIU family! "
                    "Our admission office is open now and would be happy to assist you with your application. "
                    "Would you like me to guide you to the Admission office?"
                )
                with pending_action_lock:
                    pending_action = "go_to_Admission"
                response_queue.put(response)
                logger.info(f"Responding: {response}")
            else:
                next_open_day, next_open_time = get_next_opening("Admission")
                response = (
                    "We are thrilled that you're interested in joining the GIU family! "
                    f"However, our admission office is currently closed and will reopen on {next_open_day.capitalize()} at {next_open_time}. "
                    "Would you like help with something else?"
                )
                with pending_action_lock:
                    pending_action = "ask_if_help_needed"
                response_queue.put(response)
                logger.info(f"Responding: {response}")
            return jsonify({"response": response})
        elif any(
            keyword in command
            for keyword in [
                "computer science major",
                "cs major",
                "computer science department",
                "cs department",
                "tell me about computer science",
                "i want to study computer science",
                "computer science information",
            ]
        ):
            # Provide a brief about the computer science major
            response = (
                "The Computer Science major at GIU offers a comprehensive study of computing "
                "systems and software. It covers programming, algorithms, data structures, "
                "and more. We are proud of our state-of-the-art facilities and expert faculty. "
                "Would you like to see if Dr. Nada is available to provide more information?"
            )
            with pending_action_lock:
                pending_action = "check_doctor_availability_dr_nada"
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})
        elif any(
            keyword in command
            for keyword in [
                "giu",
                "german international university",
            ]
        ):
            # General welcome message for GIU
            response = "Welcome to the German International University! How can I assist you today?"
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})

    # Step 2: Provide initial conversational response based on identified room or doctor
    if room:
        availability = check_room_availability(room)
        if availability["is_open"]:
            response = f"{room.replace('_', ' ')} is open. Would you like me to guide you there?"
            with pending_action_lock:
                pending_action = f"go_to_{room}"
            response_queue.put(response)
            logger.info(f"Responding: {response}")
        else:
            next_open_day, next_open_time = get_next_opening(room)
            response = f"{room.replace('_', ' ')} is currently closed and will open on {next_open_day.capitalize()} at {next_open_time}. Would you like help with something else?"
            with pending_action_lock:
                pending_action = "ask_if_help_needed"
            response_queue.put(response)
            logger.info(f"Responding: {response}")
        return jsonify({"response": response})

    elif doctor:
        # Handle 'now' in command to represent current day
        if "now" in command_normalized:
            day_in_text = datetime.now().strftime("%A").lower()
            schedule = get_doctor_schedule(doctor, day_in_text)
            if schedule:
                schedule_str = ", ".join(schedule)
                response = f"{doctor.replace('_', ' ').title()} is available at the following times today: {schedule_str}."
            else:
                response = f"{doctor.replace('_', ' ').title()} is not available today."
            response += " Is there anything else I can assist you with?"
            pending_action = "ask_if_help_needed"
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})
        else:
            # Extract day if specified
            day_in_text = extract_day_from_text(command_normalized)
            if day_in_text:
                schedule = get_doctor_schedule(doctor, day_in_text)
                if schedule:
                    schedule_str = ", ".join(schedule)
                    response = f"{doctor.replace('_', ' ').title()} is available at the following times on {day_in_text.capitalize()}: {schedule_str}."
                else:
                    response = f"{doctor.replace('_', ' ').title()} is not available on {day_in_text.capitalize()}."
                response += " Is there anything else I can assist you with?"
                pending_action = "ask_if_help_needed"
                response_queue.put(response)
                logger.info(f"Responding: {response}")
                return jsonify({"response": response})
            else:
                availability = get_doctor_availability_data(doctor)
                if availability["is_available"]:
                    response = f"{doctor.replace('_', ' ').title()} is available now. Would you like me to guide you to their office?"
                    with pending_action_lock:
                        pending_action = f"go_to_{doctor}"
                    logger.info(f"Responding: {response}")
                else:
                    response = f"{doctor.replace('_', ' ').title()} is not available now. {availability['next_availability']} Would you like help with something else?"
                    with pending_action_lock:
                        pending_action = "ask_if_help_needed"
                    logger.info(f"Responding: {response}")
                response_queue.put(response)
                return jsonify({"response": response})

    # Handle greetings or fallback for unrecognized commands
    elif command.lower() in ["hi", "hey", "hello"]:
        response = "Hello! Welcome to GIU's campus. How can I assist you today?"
    elif command == "kill":
        response = "Stopping the program. Goodbye!"
    else:
        response = "I'm sorry, I didn't quite understand that. Could you please rephrase your request?"

    # Put the response into the queue and log it
    response_queue.put(response)
    logger.info(f"Responding: {response}")
    return jsonify({"response": response})


def get_next_opening(room):
    """
    Helper function to find the next opening day and time for a room.

    Args:
        room (str): The name of the room.

    Returns:
        tuple: Next opening day and time as strings.
    """
    current_day = datetime.now().strftime("%A").lower()
    current_time = datetime.now().strftime("%H:%M")
    days_of_week = DAYS_OF_WEEK

    # Iterate through the next 7 days to find the next opening time
    for i in range(1, 8):  # Start from next day, check up to a week ahead
        day_index = (days_of_week.index(current_day) + i) % 7
        next_day = days_of_week[day_index]
        if next_day in weekly_schedule.get(room, {}):
            opening_time = weekly_schedule[room][next_day]["opens_at"]
            return next_day, opening_time

    return (
        None,
        None,
    )  # If no opening time is found, which is unlikely in a weekly schedule


def get_room_opening_times(room, day):
    """
    Get the opening times for a room on a specific day.

    Args:
        room (str): The name of the room.
        day (str): The day of the week.

    Returns:
        dict or None: Dictionary with 'opens_at' and 'closes_at' if available, else None.
    """
    day = day.lower()
    if room in weekly_schedule and day in weekly_schedule[room]:
        return weekly_schedule[room][day]
    else:
        return None


def get_doctor_schedule(doctor, day):
    """
    Get the schedule for a doctor on a specific day.

    Args:
        doctor (str): The doctor's identifier.
        day (str): The day of the week.

    Returns:
        list or None: List of available time slots if available, else None.
    """
    day = day.lower()
    doctor = doctor.lower()
    if doctor in doctor_availability and day in doctor_availability[doctor]:
        return doctor_availability[doctor][day]
    else:
        return None


def extract_day_from_text(text):
    """
    Extract day of the week from text.

    Args:
        text (str): The input text.

    Returns:
        str or None: The day of the week if found, else None.
    """
    text = text.lower()
    for day in DAYS_OF_WEEK:
        if day in text:
            return day
    return None


def get_doctor_availability_data(doctor_id):
    """
    Check the availability of a doctor based on the current time and schedule.

    Args:
        doctor_id (str): The doctor's identifier.

    Returns:
        dict: Availability status and next available time if not available.
    """
    current_day = datetime.now().strftime("%A").lower()
    current_time = datetime.now().strftime("%H:%M")

    availability = doctor_availability.get(doctor_id, {})
    if current_day in availability:
        for time_range in availability[current_day]:
            start_time, end_time = map(str.strip, time_range.split("-"))
            if start_time <= current_time <= end_time:
                return {"is_available": True}
        # Find next available time
        for time_range in availability[current_day]:
            start_time, _ = map(str.strip, time_range.split("-"))
            if current_time < start_time:
                next_availability = f"The next available time is today at {start_time}."
                return {"is_available": False, "next_availability": next_availability}
        # If no times left today, check next day
        next_day_index = (DAYS_OF_WEEK.index(current_day) + 1) % 7
        next_day = DAYS_OF_WEEK[next_day_index]
        if next_day in availability:
            next_time = availability[next_day][0].split("-")[0].strip()
            next_availability = (
                f"The next available time is on {next_day.capitalize()} at {next_time}."
            )
            return {"is_available": False, "next_availability": next_availability}
        else:
            return {
                "is_available": False,
                "next_availability": "No availability found.",
            }
    else:
        # Find next available day
        days_of_week = DAYS_OF_WEEK
        current_day_index = days_of_week.index(current_day)
        for i in range(1, 7):
            next_day_index = (current_day_index + i) % 7
            next_day = days_of_week[next_day_index]
            if next_day in availability:
                next_time = availability[next_day][0].split("-")[0].strip()
                next_availability = f"The next available time is on {next_day.capitalize()} at {next_time}."
                return {"is_available": False, "next_availability": next_availability}
        return {"is_available": False, "next_availability": "No availability found."}


@app.route("/")
def home():
    """Render the home page."""
    return render_template("index.html")


@app.route("/doctor_availability", methods=["GET"])
def doctor_availability_endpoint():
    """
    Fetch doctor availability information.

    Query Parameters:
        doctor_id (str): The identifier of the doctor.

    Returns:
        JSON response containing availability data or error message.
    """
    doctor_id = request.args.get("doctor_id")
    if doctor_id:
        availability = doctor_availability.get(doctor_id.lower(), {})
        if availability:
            return jsonify({"status": "success", "data": {doctor_id: availability}})
        return jsonify({"status": "error", "message": "Doctor not found"}), 404
    return jsonify({"status": "success", "data": doctor_availability})


@app.route("/command", methods=["POST"])
def handle_command():
    """
    Handles the POST request from the frontend, processes the command, and sends back a response.

    Expects JSON data with a 'text' field containing the user's command.

    Returns:
        JSON response with the assistant's reply.
    """
    data = request.json
    logger.debug(f"Received data: {data}")
    command_text = data.get("text", "").strip()

    if command_text:
        logger.debug(f"Command received: {command_text}")

        # Check if there's a pending action that requires a response
        if pending_action and (
            pending_action.startswith("go_to_")
            or pending_action == "ask_if_help_needed"
            or pending_action.startswith("check_doctor_availability_")
            or pending_action.startswith("ask_for_day_room_")
            or pending_action.startswith("ask_for_day_doctor_")
        ):
            # Pass the original command text to open_application for processing
            return open_application(command_text, command_text)

        # Submit the classification task to executor
        future = executor.submit(classify_command_cached, command_text)
        predicted_label = future.result()

        logger.info(f"Matched label: {predicted_label}")

        if predicted_label != "none":
            return open_application(predicted_label, command_text)
        else:
            # If no label matched with high confidence, treat the original text
            return open_application(command_text, command_text)

    return jsonify({"response": "No command received."})


@app.route("/get_prompt", methods=["GET"])
def get_prompt():
    """
    Return the prompt message if available.

    Returns:
        JSON response with the prompt or None if no prompt is available.
    """
    try:
        prompt = prompt_queue.get_nowait()
        return jsonify({"prompt": prompt})
    except queue.Empty:
        return jsonify({"prompt": None})


@app.route("/post_choice", methods=["POST"])
def post_choice():
    """
    Receive the user's choice and put it into command_queue.

    Expects JSON data with a 'choice' field.

    Returns:
        JSON response indicating success or error.
    """
    data = request.json
    choice = data.get("choice")
    if choice:
        command_queue.put(choice)
        return jsonify({"status": "success"})
    else:
        return jsonify({"status": "error", "message": "No choice provided."}), 400


def open_browser_after_delay(url, delay=1):
    """
    Waits for a specified delay and then opens the browser.

    Args:
        url (str): The URL to open.
        delay (int): Delay in seconds before opening the browser.
    """
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
            waypoints (list): List of waypoint coordinates.
            waypoint_names (list): List of waypoint names.
            walls (list): List of Wall objects.
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
        self.is_returning_to_start = False  # Flag for returning to start
        self.prompt_queue = prompt_queue  # Queue to communicate with frontend
        self.sensors = []
        self.create_sensors()
        self.path = []  # List of waypoints to follow
        self.arduino_obstacle_detected = False  # Flag for Arduino obstacle detection
        self.obstacle_response_sent = False  # Flag to prevent multiple responses
        self.started_moving = False  # Flag to indicate if movement has started

        # Initialize waypoint dictionary for easy lookup
        self.waypoint_dict = {
            name: position
            for name, position in zip(self.waypoint_names, self.waypoints)
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
        """Update sensor positions based on the current angle."""
        self.create_sensors()

    def check_sensors(self):
        """
        Check sensors for obstacle detection.

        Returns:
            list: List of tuples containing sensor angle and detection status.
        """
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

    def return_to_start(self):
        """
        Set the waypoints for the car to return to the starting point.
        """
        # Define the path back to start using pre-defined waypoints
        if self.current_location_name != "Start":
            path_key = (self.current_location_name, "Start")
            if path_key in self.waypoint_paths:
                self.path = [
                    self.waypoint_dict[wp_name]
                    for wp_name in self.waypoint_paths[path_key]
                ]
                self.current_target = self.path.pop(0)
                self.destination_name = "Start"
                self.moving = True
                self.is_returning_to_start = True  # Indicate return journey
                self.state_reason = "Returning to start via checkpoints"
                logger.info(
                    f"Returning to start from {self.current_location_name} following checkpoints."
                )
            else:
                logger.warning(
                    f"No return path defined from {self.current_location_name} to Start."
                )

    def line_rect_intersect(self, line, rect):
        """
        Check if a line intersects with a rectangle.

        Args:
            line (tuple): Tuple containing two points defining the line.
            rect (pygame.Rect): The rectangle to check against.

        Returns:
            bool: True if intersection occurs, False otherwise.
        """
        rect_lines = [
            ((rect.left, rect.top), (rect.right, rect.top)),  # Top edge
            ((rect.right, rect.top), (rect.right, rect.bottom)),  # Right edge
            ((rect.right, rect.bottom), (rect.left, rect.bottom)),  # Bottom edge
            ((rect.left, rect.bottom), (rect.left, rect.top)),  # Left edge
        ]
        for rect_line in rect_lines:
            if self.line_line_intersect(line, rect_line):
                return True
        return False

    def line_line_intersect(self, line1, line2):
        """
        Check if two lines intersect.

        Args:
            line1 (tuple): First line defined by two points.
            line2 (tuple): Second line defined by two points.

        Returns:
            bool: True if lines intersect, False otherwise.
        """
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
        """
        Move the car forward based on its speed and angle.
        """
        new_x = self.x + self.speed * math.cos(math.radians(self.angle))
        new_y = self.y - self.speed * math.sin(math.radians(self.angle))
        if not self.check_collision(new_x, new_y):
            self.x = new_x
            self.y = new_y
            self.update_sensors()
        else:
            logger.warning("Collision detected! Movement blocked.")

    def check_point_reached(self):
        """
        Check if the current target point has been reached.

        Returns:
            bool: True if the target is reached, False otherwise.
        """
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
                    prompt_message = f"Reached {destination_name.replace('_', ' ')}"
                    self.prompt_queue.put(prompt_message)
                    self.state_reason = "Awaiting user choice"
                else:
                    # If returning to start, reset the flag and set reason
                    self.is_returning_to_start = False
                    self.state_reason = "At Start Point"
            return True
        return False

    def get_waypoint_name(self, position):
        """
        Get the name of the waypoint given its position.

        Args:
            position (tuple): The (x, y) coordinates of the waypoint.

        Returns:
            str or None: The name of the waypoint if found, else None.
        """
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
        """
        Set a new target for the car to navigate to.

        Args:
            target_point (tuple): The (x, y) coordinates of the target waypoint.
            destination_name (str): The name of the destination.
        """
        self.destination_name = destination_name
        self.moving = True
        self.state_reason = f"Moving towards {destination_name.replace('_', ' ')}"
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
        """
        Update the car's state by rotating and moving towards the target.
        Handles obstacle detection and movement logic.
        """
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
        """
        Draw the car's current status and reason on the screen.

        Args:
            surface (pygame.Surface): The surface to draw the status on.
        """
        font = pygame.font.SysFont(None, 24)
        status = "MOVING" if self.moving else "STOPPED"

        # Render status and reason as text
        status_text = font.render(f"Robot Status: {status}", True, BLACK)
        reason_text = font.render(f"Reason: {self.state_reason}", True, BLACK)

        # Determine the width of the status text to place reason text with a gap
        status_width = status_text.get_width()
        margin = 20  # Space between status and reason texts

        # Draw status and reason side by side with a margin
        surface.blit(status_text, (10, HEIGHT + 10))
        surface.blit(reason_text, (10 + status_width + margin, HEIGHT + 10))

    def draw(self, surface):
        """
        Render the car and waypoints on the screen.

        Args:
            surface (pygame.Surface): The surface to draw on.
        """
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
        ("Admission", "Start"): [
            "M216",
            "M215",
            "Start",
        ],  # Updated path to avoid blocked route
        ("Start", "M216"): ["M215", "M216"],
        ("Start", "Admission"): [
            "M215",
            "M216",
            "Admission",
        ],  # Must go through M215 and M216
        ("M215", "Admission"): ["M216", "Admission"],
        ("M216", "M215"): ["M215"],
        ("M216", "Start"): ["M215", "Start"],
        ("Admission", "M216"): ["M216"],
        ("Admission", "M215"): ["M216", "M215"],
        ("M215", "Start"): ["Start"],
        ("Start", "dr_nada"): ["M215", "M216", "dr_nada"],
        ("dr_nada", "Start"): ["M216", "M215", "Start"],
        ("M216", "dr_nada"): ["dr_nada"],
        ("dr_nada", "M216"): ["M216"],
        # Add more paths as needed
    }


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
        self.obstacle_response_sent = False  # Flag to prevent multiple responses

    def run(self):
        """Run the thread to continuously read from the serial port."""
        try:
            # Open the serial connection
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
            (500, 300),  # dr_nada's office (Updated position)
        ]
        self.waypoint_names = [
            "Start",
            "M215",
            "M216",
            "Admission",
            "dr_nada",
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
        """
        Choose a waypoint based on mouse click and update the car's target.

        Args:
            mouse_x (int): The x-coordinate of the mouse click.
            mouse_y (int): The y-coordinate of the mouse click.
        """
        closest_index = None
        min_distance = float("inf")

        # Find the closest waypoint within a threshold distance
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
            # Send 'START_SERVO' command to Arduino to indicate movement
            self.send_command("START_SERVO")

    def send_command(self, command):
        """
        Send a command to the Arduino via serial.

        Args:
            command (str): The command string to send.
        """
        self.serial_reader.send_command(command)

    def process_commands(self):
        """
        Process any incoming commands from Flask via the command_queue.
        """
        try:
            while True:
                command = command_queue.get_nowait()
                logger.info(f"Processing command from Flask: {command}")
                if command.startswith("go_to_"):
                    # Extract the room or doctor's name dynamically
                    location = command[
                        len("go_to_") :
                    ]  # Get the substring after 'go_to_'
                    location_normalized = (
                        location.replace("-", "_").replace(" ", "_").title()
                    )

                    # Check if the location exists in waypoint_names or is a valid doctor
                    if (
                        location_normalized.lower() in VALID_DOCTORS
                        or location_normalized in self.waypoint_names
                    ):
                        target_point = self.waypoint_dict.get(
                            location_normalized.lower(),
                            self.waypoint_dict.get(location_normalized),
                        )
                        if target_point:
                            self.car.set_target(target_point, location_normalized)
                            logger.info(
                                f"Setting target to {location_normalized}: {target_point}"
                            )
                            self.send_command("START_SERVO")
                        else:
                            logger.warning(f"Unknown location: {location_normalized}")
                    else:
                        logger.warning(f"Unknown location: {location_normalized}")
                elif command == "user_choice_done":
                    # User has completed their choice, return to start
                    self.car.return_to_start()
                    response_queue.put("Goodbye, going to start point.")
                elif command == "user_choice_go_another":
                    # User wants to choose another waypoint
                    self.car.state_reason = "Waiting for waypoint"
                    response_queue.put("Where do you want to go next")
        except queue.Empty:
            pass  # No commands to process

    def process_responses(self):
        """
        Process any incoming responses from the response_queue.
        """
        try:
            while True:
                response = response_queue.get_nowait()
                logger.info(f"Processing response: {response}")
                # Handle Text-to-Speech (TTS) in a separate thread to avoid blocking
                threading.Thread(
                    target=self.perform_tts, args=(response,), daemon=True
                ).start()
        except queue.Empty:
            pass  # No responses to process

    def perform_tts(self, text):
        """
        Generate and play TTS audio for the given text.

        Args:
            text (str): The text to convert to speech.
        """
        try:
            temp_file = f"response_{uuid.uuid4()}.mp3"  # Temporary file name
            tts = gTTS(text=text, lang="en")  # Initialize gTTS
            tts.save(temp_file)  # Save the speech to a file
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
        """
        Run the Flask app and open the browser after a short delay.
        """
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

            # Update car movement based on current state and sensors
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
            self.screen.fill(WHITE)  # Clear the screen with white background
            self.draw_walls()
            self.car.draw(self.screen)
            self.car.draw_status(self.screen)  # Draw car status on the screen

            # Update the display and tick the clock
            pygame.display.flip()
            self.clock.tick(FPS)

        # Clean up and stop serial reader thread when the game loop ends
        self.serial_reader.stop()
        self.serial_reader.join()

        pygame.quit()
        sys.exit()


# -------------------- Main Block ---------------------#

if __name__ == "__main__":
    try:
        # Initialize and run the game
        game = Game()
        game.run()
    except Exception as e:
        # Log any unexpected exceptions
        logger.exception(f"An unexpected error occurred: {e}")
