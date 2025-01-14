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
import re
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import torch

# Initialize thread-safe queues for inter-thread communication
command_queue = queue.Queue()
response_queue = queue.Queue()
prompt_queue = queue.Queue()

# -------------------- Constants ---------------------#

pygame.init()
pygame.font.init()  # Ensure font module is initialized

# Window dimensions
WIDTH, HEIGHT = 600, 450  # Main simulation area: 600x450
BUTTON_AREA_HEIGHT = 150  # Bottom status area: 600x150
TOTAL_HEIGHT = HEIGHT + BUTTON_AREA_HEIGHT  # Total window size: 600x600

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED_COLOR = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
DARK_GRAY = (169, 169, 169)  # Added definition for DARK_GRAY

# Real-world dimensions (meters)
inner_points_real = [
    (0, 0),  # Start point
    (0, 20),  # First corner
    (39.3, 20),  # Second corner
    (39.3, 0),  # Last point
]

outer_points_real = [
    (-2.7, 0),  # Start point
    (-2.7, 23.5),  # First corner
    (42.8, 23.5),  # Second corner
    (42.8, 0),  # Last point
]

# Scale factor based on real-world corridor dimensions and screen size
PAD = 40  # Padding around corridor in the Pygame window


def compute_scale(width_m, height_m, screen_width, screen_height, pad):
    scale_w = (screen_width - 2 * pad) / width_m
    scale_h = (screen_height - 2 * pad) / height_m
    return min(scale_w, scale_h)


real_width = 42.8 + 2.7  # Outer width in meters = 45.5m
real_height = 23.5  # Outer height in meters

SCALE = 10  # pixels per meter (Adjusted for better visibility)

# Robot real-life diameter
ROBOT_DIAMETER_REAL = 0.3  # meters (30 cm)

# Scaled robot diameter (for collision detection)
ROBOT_DIAMETER_SCALED = ROBOT_DIAMETER_REAL * SCALE  # 0.3m * 10 = 3 pixels

# Minimum visual size for robot in Pygame
ROBOT_VISUAL_DIAMETER = max(20, int(ROBOT_DIAMETER_SCALED))  # At least 20 pixels

# Adjusted Car Speed
CAR_SPEED = 0.56  # meters per second (2 km/h)
CAR_ROTATION_SPEED = 90  # Degrees per second

# Sensor length definitions
SENSOR_LENGTH_REAL = 0.15  # 15 cm in meters
SENSOR_LENGTH = max(
    15, int(SENSOR_LENGTH_REAL * SCALE)
)  # pixels, minimum 15 for visibility

SENSOR_ANGLES = [-30, 0, 30]
WAYPOINT_THRESHOLD = 1  # pixels (increased from 2)
FPS = 30

SERIAL_PORT = "COM8"  # Update this to your Arduino's serial port
BAUD_RATE = 9600

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to reduce verbosity
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# -------------------- Flask App Setup ---------------------#

# Initialize NLP pipeline
device = 0 if torch.cuda.is_available() else -1
nlp = pipeline(
    "zero-shot-classification",
    model="microsoft/deberta-base-mnli",
    tokenizer="microsoft/deberta-base-mnli",
    framework="pt",
    device=device,
)
if device == 0:
    logger.info("NLP pipeline is using GPU.")
else:
    logger.info("NLP pipeline is using CPU.")


# Define buildings and rooms
VALID_BUILDINGS = {
    "A": {"name": "Building A", "rooms": {"A101", "A102", "A103"}},
    "M": {
        "name": "Building M",
        "rooms": {"M415", "M416", "Admission", "Financial", "Student_Affairs"},
    },
    "S": {"name": "Building S", "rooms": {"S301", "S302", "S303"}},
}
all_rooms = set()
for b_data in VALID_BUILDINGS.values():
    all_rooms.update(b_data["rooms"])

VALID_DOCTORS = ["dr_slim", "doctor_slim", "doc_slim", "dr_nada", "dr_omar"]
DAYS_OF_WEEK = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

# Define labels for zero-shot classification
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
]

# Add variations for rooms
for b_data in VALID_BUILDINGS.values():
    for room in b_data["rooms"]:
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

# Add variations for doctors
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

# Removed "now" from labels to prevent misclassification with "yes"
# labels.append("now")  # Removed to avoid confusion

labels.extend(DAYS_OF_WEEK)
labels.extend(
    [
        "giu",
        "german international university",
        "apply to giu",
        "admission at giu",
        "next semester at giu",
        "apply for next semester",
        "computer science department",
        "cs department",
    ]
)

# Weekly schedule for rooms
weekly_schedule = {
    "Financial": {
        day: {"opens_at": "09:00", "closes_at": "17:00"} for day in DAYS_OF_WEEK
    },
    "Student_Affairs": {
        day: {"opens_at": "10:00", "closes_at": "18:00"} for day in DAYS_OF_WEEK
    },
    "Admission": {
        day: {"opens_at": "00:00", "closes_at": "23:59"}
        for day in DAYS_OF_WEEK  # normal timing "opens_at": "08:00", "closes_at": "16:00" just changed to 24 hours for testing
    },
}


def create_doctor_schedule():
    schedule = {}
    common_times = [
        "08:30 - 10:00",
        "10:15 - 11:45",
        "12:00 - 13:30",
        "00:00 - 23:59",  # normal timing "15:45 - 17:15", just changed to 24 hours for testing
    ]
    for doctor in VALID_DOCTORS:
        schedule[doctor] = {day: common_times.copy() for day in DAYS_OF_WEEK}
    return schedule


doctor_availability = create_doctor_schedule()


def check_room_availability(room):
    current_day = datetime.now().strftime("%A")  # Removed .lower()
    current_time = datetime.now().strftime("%H:%M")
    if room in weekly_schedule and current_day in weekly_schedule[room]:
        opening_time = weekly_schedule[room][current_day]["opens_at"]
        closing_time = weekly_schedule[room][current_day]["closes_at"]
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
    else:
        # No specific schedule for this room, assume always open without times
        return {"is_open": True}


def get_doctor_schedule(doctor, day):
    day = day  # Assuming day is already in proper casing
    doctor = doctor.lower()
    if doctor in doctor_availability and day in doctor_availability[doctor]:
        return doctor_availability[doctor][day]
    else:
        return None


def get_next_opening(room):
    current_day = datetime.now().strftime("%A")  # Removed .lower()
    current_time = datetime.now().strftime("%H:%M")
    days_of_week = DAYS_OF_WEEK
    for i in range(1, 8):  # Check the next 7 days to cover the entire week
        try:
            day_index = (days_of_week.index(current_day) + i) % 7
            next_day = days_of_week[day_index]
            if next_day in weekly_schedule.get(room, {}):
                opening_time = weekly_schedule[room][next_day]["opens_at"]
                return next_day, opening_time
        except ValueError:
            logger.error(f"Day '{current_day}' not found in DAYS_OF_WEEK.")
            continue
    return None, None


def get_doctor_availability_data(doctor_id):
    current_day = datetime.now().strftime("%A")  # Removed .lower()
    current_time = datetime.now().strftime("%H:%M")
    logger.info(
        f"Checking availability for {doctor_id} on {current_day} at {current_time}"
    )
    availability = doctor_availability.get(doctor_id, {})

    if not availability:
        logger.warning(f"Doctor {doctor_id} not found in availability data.")
        return {
            "is_available": False,
            "next_availability": "Doctor not found or no availability data.",
        }

    # Check availability for the current day
    today_schedule = availability.get(current_day, [])
    if today_schedule:
        for time_range in today_schedule:
            start_time, end_time = map(str.strip, time_range.split("-"))
            logger.info(f"Checking time range {start_time} - {end_time} for today.")
            if start_time <= current_time <= end_time:
                logger.info(f"Doctor {doctor_id} is available now.")
                return {"is_available": True}

    logger.info(
        f"Doctor {doctor_id} is not available today. Searching for next available day."
    )

    # If not available today, find the next available day
    for i in range(1, 8):  # Check the next 7 days to cover the entire week
        try:
            next_day_index = (DAYS_OF_WEEK.index(current_day) + i) % 7
            next_day = DAYS_OF_WEEK[next_day_index]
            next_day_schedule = availability.get(next_day, [])
            if next_day_schedule:
                # Assuming the earliest available time is the first time range
                next_time = next_day_schedule[0].split("-")[0].strip()
                next_availability = (
                    f"The next available time is on {next_day} at {next_time}."
                )
                logger.info(
                    f"Doctor {doctor_id} will be available on {next_day} at {next_time}."
                )
                return {"is_available": False, "next_availability": next_availability}
        except ValueError:
            logger.error(f"Day '{current_day}' not found in DAYS_OF_WEEK.")
            continue

    # If no availability found in the next week
    logger.warning(
        f"No availability found for doctor {doctor_id} in the upcoming week."
    )
    return {
        "is_available": False,
        "next_availability": "No availability found in the upcoming week.",
    }


app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Initialize pending_action with thread safety
pending_action = None
pending_action_lock = Lock()
executor = ThreadPoolExecutor(max_workers=4)


@lru_cache(maxsize=128)
def classify_command_cached(command_text):
    overall_start_time = time.perf_counter()  # Start overall timing
    logger.info(f"[Classification] Starting classification for: '{command_text}'")

    # Start timing for NLP pipeline
    pipeline_start_time = time.perf_counter()
    try:
        result = nlp(
            command_text,
            candidate_labels=tuple(labels),
            hypothesis_template="This text is about {}.",
            multi_label=True,
        )
        pipeline_end_time = time.perf_counter()
        pipeline_elapsed = pipeline_end_time - pipeline_start_time
        logger.info(
            f"[Classification] NLP pipeline processing took {pipeline_elapsed:.4f} seconds"
        )
    except Exception as e:
        pipeline_end_time = time.perf_counter()
        pipeline_elapsed = pipeline_end_time - pipeline_start_time
        logger.error(
            f"[Classification] Error during NLP pipeline processing: {e} (took {pipeline_elapsed:.4f} seconds)"
        )
        return "none"

    # Start timing for post-processing
    post_start_time = time.perf_counter()
    confidence_threshold = 0.3
    matched_labels = [
        label
        for label, score in zip(result["labels"], result["scores"])
        if score > confidence_threshold
    ]
    post_end_time = time.perf_counter()
    post_elapsed = post_end_time - post_start_time
    logger.info(f"[Classification] Post-processing took {post_elapsed:.4f} seconds")

    overall_end_time = time.perf_counter()
    overall_elapsed = overall_end_time - overall_start_time
    logger.info(
        f"[Classification] Overall classification took {overall_elapsed:.4f} seconds for command: '{command_text}'"
    )

    return matched_labels[0] if matched_labels else "none"


def is_affirmative(response):
    affirmative_responses = [
        "yes",
        "yeah",
        "yep",
        "sure",
        "please do",
        "of course",
        "affirmative",
        "i'm done",
        "done",
    ]
    return any(word in response.lower() for word in affirmative_responses)


def is_negative(response):
    negative_responses = [
        "no",
        "nope",
        "not now",
        "maybe later",
        "negative",
        "i need something else",
    ]
    return any(word in response.lower() for word in negative_responses)


def extract_day_from_text(text):
    text = text.lower()
    for day in DAYS_OF_WEEK:
        if day.lower() in text:
            return day  # Return with original casing
    return None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/doctor_availability", methods=["GET"])
def doctor_availability_endpoint():
    doctor_id = request.args.get("doctor_id")

    if doctor_id:
        # Normalize the doctor_id to match the keys in `doctor_availability`
        normalized_id = (
            f"dr_{doctor_id.lower()}"
            if not doctor_id.startswith("dr_")
            else doctor_id.lower()
        )

        # Check if the doctor exists in the availability data
        availability = doctor_availability.get(normalized_id, {})
        if availability:
            return jsonify({"status": "success", "data": {normalized_id: availability}})

        # If doctor is not found
        return jsonify({"status": "error", "message": "Doctor not found"}), 404

    # If no doctor_id is provided, return all availability
    return jsonify({"status": "success", "data": doctor_availability})


@app.route("/command", methods=["POST"])
def handle_command():
    start_time = time.perf_counter()  # Start timing
    data = request.json
    logger.info(f"Received data: {data}")
    command_text = data.get("text", "").strip()

    if command_text:
        logger.info(f"Command received: {command_text}")
        global pending_action
        with pending_action_lock:
            current_pending = pending_action

        if current_pending and (
            current_pending.startswith("go_to_")
            or current_pending == "ask_if_help_needed"
            or current_pending.startswith("check_doctor_availability_")
            or current_pending.startswith("ask_for_day_room_")
            or current_pending.startswith("ask_for_day_doctor_")
        ):
            response = open_application(current_pending, command_text)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            logger.info(f"handle_command took {elapsed_time:.4f} seconds")
            return response  # Return the Response object directly

        # Classify the command asynchronously
        future = executor.submit(classify_command_cached, command_text)
        predicted_label = future.result()
        logger.info(f"Matched label: {predicted_label}")

        response = open_application(predicted_label, command_text)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.info(f"handle_command took {elapsed_time:.4f} seconds")

        return response  # Return the Response object directly

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    logger.info(f"handle_command took {elapsed_time:.4f} seconds")
    return jsonify({"response": "No command received."})


@app.route("/get_prompt", methods=["GET"])
def get_prompt():
    try:
        prompt = prompt_queue.get_nowait()
        return jsonify({"prompt": prompt})
    except queue.Empty:
        return jsonify({"prompt": None})


@app.route("/user_choice", methods=["POST"])
def handle_user_choice():
    data = request.json
    choice = data.get("choice")
    logger.info(f"User made a choice: {choice}")
    if choice.lower() in ["i'm done", "done"]:
        command_queue.put("user_choice_done")
        response = "Goodbye, going to start point."
    elif choice.lower() in ["need something else", "another"]:
        command_queue.put("user_choice_another")
        response = "How may I help you further?"
    else:
        response = "Invalid choice."
    return jsonify({"response": response})


@app.route("/robot_status", methods=["GET"])
def robot_status():
    # We assume `game` is a module-level variable or a global we can access
    is_at_start = game.car.current_location_name.lower() == "start"
    return jsonify(
        {"current_location": game.car.current_location_name, "is_at_start": is_at_start}
    )


def open_application(command, original_command_text):
    """
    Main logic for handling commands.
    """
    global pending_action
    response = ""
    command = command.strip().lower()

    logger.info(f"open_application called with command: {command}")
    logger.info(f"Original command text: {original_command_text}")

    # =================== Handling user confirmations (yes/no) ===================
    with pending_action_lock:
        current_pending = pending_action

    if current_pending and current_pending.startswith("go_to_"):
        if is_affirmative(original_command_text):
            location = current_pending[len("go_to_") :]
            location_normalized = location.replace("-", "_").replace(" ", "_").lower()
            logger.info(f"User confirmed to go to location: {location_normalized}")
            command_queue.put(f"go_to_{location_normalized}")
            if location_normalized.lower() in VALID_DOCTORS:
                response = f"Taking you to {location_normalized.replace('_', ' ')}'s office now."
            else:
                response = (
                    f"Taking you to the {location_normalized.replace('_', ' ')} now."
                )
            with pending_action_lock:
                pending_action = None  # Reset pending_action after handling
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})
        elif is_negative(original_command_text):
            response = "Okay, let me know if you need anything else."
            with pending_action_lock:
                pending_action = None  # **Clear the pending_action**
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})
        else:
            response = "I'm sorry, I didn't catch that. Please say yes or no."
            # Do NOT clear pending_action to continue awaiting a valid response
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})

    elif current_pending == "ask_if_help_needed":
        if is_affirmative(original_command_text):
            response = "Great! What would you like help with?"
            with pending_action_lock:
                pending_action = None
        elif is_negative(original_command_text):
            response = "Okay, feel free to ask if you need any assistance. Goodbye!"
            with pending_action_lock:
                pending_action = None  # **Clear the pending_action**
        else:
            response = "I'm sorry, I didn't catch that. Please say yes or no."
            # Do NOT clear pending_action to continue awaiting a valid response
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})
        response_queue.put(response)
        logger.info(f"Responding: {response}")
        return jsonify({"response": response})

    elif current_pending and current_pending.startswith("check_doctor_availability_"):
        if is_affirmative(original_command_text):
            doctor_id = current_pending[len("check_doctor_availability_") :].replace(
                "-", "_"
            )
            availability = get_doctor_availability_data(doctor_id)
            if availability["is_available"]:
                response = f"{doctor_id.replace('_', ' ').title()} is available now. Would you like me to guide you to their office?"
                with pending_action_lock:
                    pending_action = f"go_to_{doctor_id}"
            else:
                response = f"{doctor_id.replace('_', ' ').title()} is not available now. {availability['next_availability']} Would you like help with something else?"
                with pending_action_lock:
                    pending_action = "ask_if_help_needed"
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})
        elif is_negative(original_command_text):
            response = "Okay, let me know if you need anything else."
            with pending_action_lock:
                pending_action = None  # **Clear the pending_action**
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})
        else:
            response = "I'm sorry, I didn't catch that. Please say yes or no."
            # Do NOT clear pending_action to continue awaiting a valid response
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})

    elif current_pending and current_pending.startswith("ask_for_day_room_"):
        room = current_pending[len("ask_for_day_room_") :]
        if is_negative(original_command_text):
            response = "Okay, let me know if you need anything else."
            with pending_action_lock:
                pending_action = None  # **Clear the pending_action**
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})

        day_in_text = extract_day_from_text(original_command_text)
        if day_in_text:
            opening_times = check_room_availability(room)
            if opening_times:
                if opening_times["is_open"]:
                    # **Modified to conditionally include times**
                    if "opens_at" in opening_times and "closes_at" in opening_times:
                        response = f"The {room.replace('_', ' ')} is open from {opening_times['opens_at']} to {opening_times['closes_at']} on {day_in_text.capitalize()}."
                    else:
                        response = f"The {room.replace('_', ' ')} is open on {day_in_text.capitalize()}."
                else:
                    # **Modified to conditionally include opening time**
                    if "opens_at" in opening_times:
                        response = f"The {room.replace('_', ' ')} is closed on {day_in_text.capitalize()} and will open next at {opening_times['opens_at']}."
                    else:
                        response = f"The {room.replace('_', ' ')} is currently closed on {day_in_text.capitalize()}."
                response += " Is there anything else I can assist you with?"
                with pending_action_lock:
                    pending_action = "ask_if_help_needed"
                response_queue.put(response)
                logger.info(f"Responding: {response}")
                return jsonify({"response": response})
            else:
                response = f"The {room.replace('_', ' ')} has no available information for {day_in_text.capitalize()}."
                with pending_action_lock:
                    pending_action = "ask_if_help_needed"
                response_queue.put(response)
                logger.info(f"Responding: {response}")
                return jsonify({"response": response})
        else:
            response = (
                "I'm sorry, I didn't catch the day. Please specify a day of the week."
            )
            # Do NOT clear pending_action to continue awaiting a valid response
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})

    elif current_pending and current_pending.startswith("ask_for_day_doctor_"):
        doctor = current_pending[len("ask_for_day_doctor_") :]
        if is_negative(original_command_text):
            response = "Okay, let me know if you need anything else."
            with pending_action_lock:
                pending_action = None  # **Clear the pending_action**
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})

        day_in_text = extract_day_from_text(original_command_text)
        if day_in_text:
            schedule = get_doctor_schedule(doctor, day_in_text)
            if schedule:
                schedule_str = ", ".join(schedule)
                response = (
                    f"{doctor.replace('_', ' ').title()} is available at the following times "
                    f"on {day_in_text}: {schedule_str}."
                )
            else:
                response = f"{doctor.replace('_', ' ').title()} is not available on {day_in_text.capitalize()}."
            response += " Would you like me to guide you to their office?"
            with pending_action_lock:
                pending_action = f"go_to_{doctor}"
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})
        else:
            response = (
                "I'm sorry, I didn't catch the day. Please specify a day of the week."
            )
            # Do NOT clear pending_action to continue awaiting a valid response
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})

    # =================== Handling Availability Queries ===================
    # Normalize the command for processing
    command_normalized = command.replace("dr ", "").replace("doctor ", "").strip()
    availability_query_match = re.search(
        r"(is|are|when|what time)(.*?)(open|close|available)", command_normalized
    )

    if availability_query_match:
        subject = availability_query_match.group(2).strip()
        logger.info(f"Subject extracted for availability query: {subject}")
        subject_normalized = subject.replace("dr ", "").replace("doctor ", "").strip()
        day_in_text = extract_day_from_text(command_normalized)
        logger.info(f"Day extracted from query: {day_in_text}")

        # Check if subject is a room
        found_room = False
        for b_data in VALID_BUILDINGS.values():
            for rm in b_data["rooms"]:
                rm_lower = rm.replace("_", " ").lower()
                if rm_lower == subject_normalized or subject_normalized.endswith(
                    rm_lower
                ):
                    found_room = True
                    if day_in_text:
                        opening_times = check_room_availability(rm)
                        if opening_times["is_open"]:
                            # **Modified to conditionally include times**
                            if (
                                "opens_at" in opening_times
                                and "closes_at" in opening_times
                            ):
                                response = f"The {rm.replace('_', ' ')} is open from {opening_times['opens_at']} to {opening_times['closes_at']} today."
                            else:
                                response = f"The {rm.replace('_', ' ')} is open today."
                        else:
                            # **Modified to conditionally include opening time**
                            if "opens_at" in opening_times:
                                response = f"The {rm.replace('_', ' ')} is closed today and will open tomorrow at {opening_times['opens_at']}."
                            else:
                                response = f"The {rm.replace('_', ' ')} is currently closed today."
                        response += " Is there anything else I can assist you with?"
                        with pending_action_lock:
                            pending_action = "ask_if_help_needed"
                        response_queue.put(response)
                        logger.info(f"Responding: {response}")
                        return jsonify({"response": response})
        if not found_room:
            # Check if subject is a doctor
            doctor = None
            for doc in VALID_DOCTORS:
                doctor_lower = doc.replace("_", " ").lower()
                doctor_normalized = (
                    doctor_lower.replace("dr ", "").replace("doctor ", "").strip()
                )
                if doctor_normalized in command_normalized:
                    doctor = doc
                    logger.info(f"Identified doctor: {doctor}")
                    break

        # Topics if no room/doctor found
        if not found_room and not doctor:
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
                availability = check_room_availability("Admission")
                if availability["is_open"]:
                    response = (
                        "We are thrilled that you're interested in joining the GIU family! "
                        "Our admission office is open now and would be happy to assist you with your application. "
                        "Would you like me to guide you to the Admission office?"
                    )
                    with pending_action_lock:
                        pending_action = "go_to_admission"
                    response_queue.put(response)
                    logger.info(f"Responding: {response}")
                else:
                    next_open_day, next_open_time = get_next_opening("Admission")
                    if next_open_day and next_open_time:
                        response = (
                            "We are thrilled that you're interested in joining the GIU family! "
                            f"However, our admission office is currently closed and will reopen on {next_open_day.capitalize()} at {next_open_time}. "
                            "Would you like help with something else?"
                        )
                    else:
                        response = (
                            "We are thrilled that you're interested in joining the GIU family! "
                            "However, our admission office is currently closed. Would you like help with something else?"
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
                response = "Welcome to the German International University! How can I assist you today?"
                response_queue.put(response)
                logger.info(f"Responding: {response}")
                return jsonify({"response": response})

    # Identify rooms or doctors for navigation
    room = None
    doctor = None

    # Check for room
    for b_data in VALID_BUILDINGS.values():
        for rm in b_data["rooms"]:
            if rm.replace("_", " ").lower() in command:
                room = rm
                logger.info(f"Identified room: {room}")
                break
        if room:
            break

    # If no room, check doctor
    if not room:
        for valid_doctor in VALID_DOCTORS:
            doctor_lower = valid_doctor.replace("_", " ").lower()
            doctor_normalized = (
                doctor_lower.replace("dr ", "").replace("doctor ", "").strip()
            )
            if doctor_normalized in command_normalized:
                doctor = valid_doctor
                logger.info(f"Identified doctor: {doctor}")
                break

    # Topics if no room/doctor found
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
            availability = check_room_availability("Admission")
            if availability["is_open"]:
                response = (
                    "We are thrilled that you're interested in joining the GIU family! "
                    "Our admission office is open now and would be happy to assist you with your application. "
                    "Would you like me to guide you to the Admission office?"
                )
                with pending_action_lock:
                    pending_action = "go_to_admission"
                response_queue.put(response)
                logger.info(f"Responding: {response}")
            else:
                next_open_day, next_open_time = get_next_opening("Admission")
                if next_open_day and next_open_time:
                    response = (
                        "We are thrilled that you're interested in joining the GIU family! "
                        f"However, our admission office is currently closed and will reopen on {next_open_day.capitalize()} at {next_open_time}. "
                        "Would you like help with something else?"
                    )
                else:
                    response = (
                        "We are thrilled that you're interested in joining the GIU family! "
                        "However, our admission office is currently closed. Would you like help with something else?"
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
            response = "Welcome to the German International University! How can I assist you today?"
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})

    # Provide response based on identified room or doctor
    if room:
        availability = check_room_availability(room)
        if availability["is_open"]:
            # **Modified to conditionally include times**
            if "opens_at" in availability and "closes_at" in availability:
                response = f"The {room.replace('_', ' ')} is open from {availability['opens_at']} to {availability['closes_at']}."
            else:
                response = f"The {room.replace('_', ' ')} is open."
            response += " Would you like me to guide you there?"
            with pending_action_lock:
                pending_action = f"go_to_{room}"
            response_queue.put(response)
            logger.info(f"Responding: {response}")
        else:
            # **Modified to conditionally include opening time**
            if "opens_at" in availability:
                response = f"The {room.replace('_', ' ')} is currently closed and will open next at {availability['opens_at']}."
            else:
                response = f"The {room.replace('_', ' ')} is currently closed."
            response += " Would you like help with something else?"
            with pending_action_lock:
                pending_action = "ask_if_help_needed"
            response_queue.put(response)
            logger.info(f"Responding: {response}")
        return jsonify({"response": response})

    elif doctor:
        if "now" in command_normalized:
            day_in_text = datetime.now().strftime("%A")
            schedule = get_doctor_schedule(doctor, day_in_text)
            if schedule:
                schedule_str = ", ".join(schedule)
                response = f"{doctor.replace('_', ' ').title()} is available at the following times today: {schedule_str}."
            else:
                response = f"{doctor.replace('_', ' ').title()} is not available today."
            response += " Would you like me to guide you to their office?"
            with pending_action_lock:
                pending_action = f"go_to_{doctor}"
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})
        else:
            day_in_text = extract_day_from_text(command_normalized)
            if day_in_text:
                schedule = get_doctor_schedule(doctor, day_in_text)
                if schedule:
                    schedule_str = ", ".join(schedule)
                    response = (
                        f"{doctor.replace('_', ' ').title()} is available at the following times "
                        f"on {day_in_text}: {schedule_str}."
                    )
                else:
                    response = f"{doctor.replace('_', ' ').title()} is not available on {day_in_text.capitalize()}."
                response += " Would you like me to guide you to their office?"
                with pending_action_lock:
                    pending_action = f"go_to_{doctor}"
                response_queue.put(response)
                logger.info(f"Responding: {response}")
                return jsonify({"response": response})
            else:
                availability = get_doctor_availability_data(doctor)
                if availability["is_available"]:
                    response = (
                        f"{doctor.replace('_', ' ').title()} is available now. "
                        "Would you like me to guide you to their office?"
                    )
                    with pending_action_lock:
                        pending_action = f"go_to_{doctor}"
                    logger.info(f"Responding: {response}")
                else:
                    response = (
                        f"{doctor.replace('_', ' ').title()} is not available now. "
                        f"{availability['next_availability']} Would you like help with something else?"
                    )
                    with pending_action_lock:
                        pending_action = "ask_if_help_needed"
                    logger.info(f"Responding: {response}")
                response_queue.put(response)
                return jsonify({"response": response})

    elif command.lower() in ["hi", "hey", "hello"]:
        response = "Hello! Welcome to GIU's campus. How can I assist you today?"
    elif command == "kill":
        response = "Stopping the program. Goodbye!"
    else:
        response = "I'm sorry, I didn't quite understand that. Could you please rephrase your request?"

    response_queue.put(response)
    logger.info(f"Responding: {response}")
    return jsonify({"response": response})


# -------------------- Robot and Pygame Setup ---------------------#


class Wall:
    """
    A 'Wall' is represented by a pygame.Rect and a mask for collision.
    """

    def __init__(self, rect):
        self.rect = rect
        self.mask = self.create_mask()

    def create_mask(self):
        wall_surface = pygame.Surface(
            (self.rect.width, self.rect.height), pygame.SRCALPHA
        )
        wall_surface.fill(BLACK)
        return pygame.mask.from_surface(wall_surface)

    def draw(self, surface, camera, color=BLACK):
        # Apply camera transformation to wall rectangle
        transformed_rect = pygame.Rect(
            camera.apply((self.rect.left, self.rect.top)),
            (
                int(self.rect.width * camera.zoom),
                int(self.rect.height * camera.zoom),
            ),
        )
        pygame.draw.rect(surface, color, transformed_rect)


class Camera:
    def __init__(self, width, height, zoom=1.0):
        self.width = width
        self.height = height
        self.zoom = zoom
        self.position = pygame.math.Vector2(0, 0)  # Camera center in world coordinates

    def apply(self, pos):
        """
        Transforms world coordinates to screen coordinates.
        """
        x, y = pos
        screen_x = (x - self.position.x) * self.zoom + self.width / 2
        screen_y = (y - self.position.y) * self.zoom + self.height / 2
        return (int(screen_x), int(screen_y))

    def update(self, target_pos):
        """
        Update camera position to follow the target, with boundary constraints.
        """
        new_pos = pygame.math.Vector2(target_pos)

        # Define boundaries based on corridor size
        corridor_left = PAD
        corridor_right = PAD + int(real_width * SCALE)
        corridor_top = PAD
        corridor_bottom = PAD + int(real_height * SCALE)

        # Calculate camera boundaries (half screen size divided by zoom)
        half_width = self.width / 2 / self.zoom
        half_height = self.height / 2 / self.zoom

        # Clamp camera position to prevent viewing outside the corridor
        new_pos.x = max(
            corridor_left + half_width, min(new_pos.x, corridor_right - half_width)
        )
        new_pos.y = max(
            corridor_top + half_height, min(new_pos.y, corridor_bottom - half_height)
        )

        self.position = new_pos

    def set_zoom(self, new_zoom):
        """
        Set a new zoom level.
        """
        self.zoom = new_zoom


class CarRobot:

    def __init__(
        self,
        x,
        y,
        waypoints,
        waypoint_names,
        walls,
        prompt_queue,
        camera,
        waypoints_real,  # New parameter
        waypoint_real_dict,  # New parameter
    ):
        self.start_x = float(x)
        self.start_y = float(y)
        self.x = float(x)
        self.y = float(y)
        self.angle = 0
        self.speed = CAR_SPEED  # m/s
        self.waypoints = waypoints
        self.waypoint_names = waypoint_names
        self.waypoints_real = waypoints_real  # Store real-world coordinates
        self.waypoint_real_dict = waypoint_real_dict  # Store real-world mapping
        self.current_target = None
        self.current_location_name = "start"
        self.destination_name = None
        self.moving = False
        self.threshold = WAYPOINT_THRESHOLD
        self.walls = walls
        self.state_reason = "Waiting for waypoint"
        self.is_returning_to_start = False
        self.prompt_queue = prompt_queue
        self.camera = camera  # Reference to the camera
        self.next_waypoint = None  # Add this line
        self.sensors = []
        self.path = []
        self.arduino_obstacle_detected = False
        self.obstacle_response_sent = False
        self.started_moving = False
        self.waypoint_dict = {
            name: position
            for name, position in zip(self.waypoint_names, self.waypoints)
        }
        self.angle_diff = 0.0
        self.previous_waypoint_real = self.waypoint_real_dict[
            "start"
        ]  # Initialize previous waypoint
        logger.info(f"Waypoint Dictionary Initialized: {self.waypoint_dict}")
        self.create_sensors()  # Now called after setting self.camera

        # Initialize last update time for time-based movement
        self.last_update_time = pygame.time.get_ticks()

        # Initialize movement trail
        self.previous_positions = []

        self.serial_reader = None

        # Define waypoint paths
        # --- ADDED FOR DEBUG: path logs
        self.waypoint_paths = {
            # Existing Paths
            ("start", "m415"): ["m415"],
            ("m415", "m416"): ["m416"],
            ("m415", "start"): ["start"],
            ("m416", "admission"): ["admission"],
            ("admission", "dr_nada"): ["dr_nada"],
            ("dr_nada", "dr_omar"): ["dr_omar"],
            ("dr_omar", "start"): ["dr_nada", "admission", "m416", "m415", "start"],
            ("start", "m416"): ["m415", "m416"],
            ("start", "admission"): ["m415", "m416", "admission"],
            ("m415", "admission"): ["m416", "admission"],
            ("admission", "m415"): ["m416", "m415"],
            ("m416", "m415"): ["m415"],
            ("m416", "start"): ["m415", "start"],
            ("admission", "start"): ["m416", "m415", "start"],
            ("start", "dr_nada"): ["m415", "m416", "admission", "dr_nada"],
            ("dr_nada", "start"): ["admission", "m416", "m415", "start"],
            ("m416", "dr_nada"): ["admission", "dr_nada"],
            ("dr_nada", "m416"): ["admission", "m416"],
            ("start", "dr_omar"): ["m415", "m416", "admission", "dr_nada", "dr_omar"],
            ("dr_omar", "start"): ["dr_nada", "admission", "m416", "m415", "start"],
            ("dr_nada", "dr_slim"): ["dr_slim"],
            ("dr_slim", "dr_nada"): ["right_corner", "dr_omar", "dr_nada"],
            ("dr_slim", "dr_omar"): ["right_corner", "dr_omar"],
            ("dr_omar", "dr_slim"): ["right_corner", "dr_slim"],
            ("start", "dr_slim"): [
                "m415",
                "m416",
                "admission",
                "dr_nada",
                "dr_omar",
                "right_corner",
                "dr_slim",
            ],
            ("dr_slim", "start"): [
                "right_corner",
                "dr_omar",
                "dr_nada",
                "admission",
                "m416",
                "m415",
                "start",
            ],
            # Missing Paths Added
            # 1. start ↔ right_corner
            ("start", "right_corner"): [
                "m415",
                "m416",
                "admission",
                "dr_nada",
                "dr_omar",
                "right_corner",
            ],
            ("right_corner", "start"): [
                "dr_omar",
                "dr_nada",
                "admission",
                "m416",
                "m415",
                "start",
            ],
            # 2. m415 ↔ dr_nada
            ("m415", "dr_nada"): ["m416", "admission", "dr_nada"],
            ("dr_nada", "m415"): ["admission", "m416", "m415"],
            # 3. m415 ↔ dr_omar
            ("m415", "dr_omar"): ["m416", "admission", "dr_nada", "dr_omar"],
            ("dr_omar", "m415"): ["dr_nada", "admission", "m416", "m415"],
            # 4. m415 ↔ right_corner
            ("m415", "right_corner"): [
                "m416",
                "admission",
                "dr_nada",
                "dr_omar",
                "right_corner",
            ],
            ("right_corner", "m415"): [
                "dr_omar",
                "dr_nada",
                "admission",
                "m416",
                "m415",
            ],
            # 5. m415 ↔ dr_slim
            ("m415", "dr_slim"): [
                "m416",
                "admission",
                "dr_nada",
                "dr_omar",
                "right_corner",
                "dr_slim",
            ],
            ("dr_slim", "m415"): [
                "right_corner",
                "dr_omar",
                "dr_nada",
                "admission",
                "m416",
                "m415",
            ],
            # 6. m416 ↔ dr_omar
            ("m416", "dr_omar"): ["admission", "dr_nada", "dr_omar"],
            ("dr_omar", "m416"): ["dr_nada", "admission", "m416"],
            # 7. m416 ↔ right_corner
            ("m416", "right_corner"): [
                "admission",
                "dr_nada",
                "dr_omar",
                "right_corner",
            ],
            ("right_corner", "m416"): ["dr_omar", "dr_nada", "admission", "m416"],
            # 8. m416 ↔ dr_slim
            ("m416", "dr_slim"): [
                "admission",
                "dr_nada",
                "dr_omar",
                "right_corner",
                "dr_slim",
            ],
            ("dr_slim", "m416"): [
                "right_corner",
                "dr_omar",
                "dr_nada",
                "admission",
                "m416",
            ],
            # 9. admission ↔ dr_omar
            ("admission", "dr_omar"): ["dr_nada", "dr_omar"],
            ("dr_omar", "admission"): ["dr_nada", "admission"],
            # 10. admission ↔ right_corner
            ("admission", "right_corner"): ["dr_nada", "dr_omar", "right_corner"],
            ("right_corner", "admission"): ["dr_omar", "dr_nada", "admission"],
            # 11. admission ↔ dr_slim
            ("admission", "dr_slim"): ["dr_nada", "dr_omar", "right_corner", "dr_slim"],
            ("dr_slim", "admission"): [
                "right_corner",
                "dr_omar",
                "dr_nada",
                "admission",
            ],
            # 12. dr_nada ↔ dr_omar
            ("dr_nada", "dr_omar"): ["dr_omar"],
            ("dr_omar", "dr_nada"): ["dr_nada"],
            # 13. dr_nada ↔ right_corner
            ("dr_nada", "right_corner"): ["dr_omar", "right_corner"],
            ("right_corner", "dr_nada"): ["dr_omar", "dr_nada"],
            # 14. dr_omar ↔ right_corner
            ("dr_omar", "right_corner"): ["right_corner"],
            ("right_corner", "dr_omar"): ["dr_omar"],
            # 15. dr_slim ↔ dr_omar
            ("dr_slim", "dr_omar"): ["right_corner", "dr_omar"],
            ("dr_omar", "dr_slim"): ["right_corner", "dr_slim"],
            # 16. dr_slim ↔ dr_nada
            ("dr_slim", "dr_nada"): ["right_corner", "dr_omar", "dr_nada"],
            ("dr_nada", "dr_slim"): ["dr_slim"],
        }

    def create_sensors(self):
        self.sensors = []
        for angle_offset in SENSOR_ANGLES:
            sensor_angle = (self.angle + angle_offset) % 360
            sensor_length = SENSOR_LENGTH_REAL * self.camera.zoom * SCALE
            sx = self.x + sensor_length * math.cos(math.radians(sensor_angle))
            sy = self.y - sensor_length * math.sin(math.radians(sensor_angle))
            self.sensors.append((sensor_angle, (sx, sy)))

    def return_to_start(self):
        # Just a helper method: set your target to the "start" waypoint.
        start_position = self.waypoint_dict["start"]  # (x, y) for the "start" waypoint
        self.set_target(start_position, "start")
        self.is_returning_to_start = True
        # Optionally update state_reason if you wish:
        self.state_reason = "Returning to start"

    def get_target_angle(self):
        if not self.current_target:
            logger.debug("No current target. Returning current angle.")
            return self.angle
        target_x, target_y = self.current_target
        dx = target_x - self.x
        dy = self.y - target_y  # Inverted Y-axis for Pygame
        angle = math.degrees(math.atan2(dy, dx))
        angle = angle % 360
        logger.debug(
            f"Calculating target angle: dx={dx}, dy={dy}, raw_angle={math.degrees(math.atan2(dy, dx)):.2f}°, final_angle={angle:.2f}°"
        )
        return angle

    def update_sensors(self):
        self.create_sensors()

    def check_sensors(self):
        sensor_data = []
        for sensor_angle, (sx, sy) in self.sensors:
            line_segment = ((self.x, self.y), (sx, sy))
            obstacle_detected = False
            for wall in self.walls:
                if self.line_rect_intersect(line_segment, wall.rect):
                    obstacle_detected = True
                    break
            sensor_data.append((sensor_angle, obstacle_detected))
        return sensor_data

    def line_rect_intersect(self, line, rect):
        (x1, y1), (x2, y2) = line
        rect_lines = [
            ((rect.left, rect.top), (rect.right, rect.top)),
            ((rect.right, rect.top), (rect.right, rect.bottom)),
            ((rect.right, rect.bottom), (rect.left, rect.bottom)),
            ((rect.left, rect.bottom), (rect.left, rect.top)),
        ]
        for r_line in rect_lines:
            if self.line_line_intersect(line, r_line):
                return True
        return False

    def line_line_intersect(self, line1, line2):
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denom == 0:
            return False
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
        return 0 <= ua <= 1 and 0 <= ub <= 1

    def get_angle_to_next_waypoint(self):
        if not self.next_waypoint:
            logger.debug("No next waypoint. Returning current angle.")
            return self.angle
        target_x, target_y = self.next_waypoint
        dx = target_x - self.x
        dy = self.y - target_y  # Inverted Y-axis for Pygame
        angle = math.degrees(math.atan2(dy, dx))
        angle = angle % 360
        logger.debug(
            f"Calculating angle to next waypoint: dx={dx}, dy={dy}, raw_angle={math.degrees(math.atan2(dy, dx)):.2f}°, final_angle={angle:.2f}°"
        )
        return angle

    def compute_single_turn(self, full_waypoints):
        """
        Given a list of waypoints (including the robot's *current position* as first),
        returns a dict:
        {
            "turn_angle": float,              # first non-trivial angle difference
            "turn_distance_from_start": float # how many meters from the *start* to that turn
        }
        If no turn is found, set them to 0.0
        """

        if len(full_waypoints) < 3:
            # Not enough segments to detect a turn
            return {"turn_angle": 0.0, "turn_distance_from_start": 0.0}

        # 1) Build arrays for angles and cumulative distances
        segment_angles = []
        cumul_dist = [
            0.0
        ]  # cumul_dist[i] = distance from the 0th waypoint up to i-th waypoint

        for i in range(len(full_waypoints) - 1):
            A = full_waypoints[i]
            B = full_waypoints[i + 1]

            # distance from A to B in *meters*
            dist_pixels = math.hypot(B[0] - A[0], B[1] - A[1])
            dist_meters = dist_pixels / SCALE
            cumul_dist.append(cumul_dist[-1] + dist_meters)

            # angle for the segment i
            dx = B[0] - A[0]
            dy = A[1] - B[1]  # because Pygame's Y is inverted
            seg_angle = math.degrees(math.atan2(dy, dx)) % 360
            segment_angles.append(seg_angle)

        # 2) Find the first non-trivial turn
        #    A turn is the difference between consecutive segment angles
        for i in range(len(segment_angles) - 1):
            angle1 = segment_angles[i]
            angle2 = segment_angles[i + 1]
            diff = (angle2 - angle1 + 360) % 360
            if diff > 180:
                diff -= 360
            # if magnitude is > ~1°, we consider it a turn
            if abs(diff) > 1.0:
                turn_angle = diff
                # The turn occurs at waypoint index (i+1),
                # so distance is cumul_dist[i+1].
                turn_dist = cumul_dist[i + 1]
                return {"turn_angle": turn_angle, "turn_distance_from_start": turn_dist}

        # If no turn found:
        return {"turn_angle": 0.0, "turn_distance_from_start": 0.0}

    def set_next_waypoint(self):
        if self.path:
            self.next_waypoint = self.path[0]
            next_wp_name = self.get_waypoint_name(self.next_waypoint)
            logger.debug(
                f"Next waypoint set to {next_wp_name} at position {self.next_waypoint}"
            )
        else:
            self.next_waypoint = None
            logger.debug("No next waypoint. Current target is the final destination.")

    def rotate(self, angle_change):
        original_angle = self.angle
        self.angle = (self.angle + angle_change) % 360
        logger.debug(
            f"Rotating: Original angle={original_angle:.2f}°, Angle change={angle_change:.2f}°, New angle={self.angle:.2f}°"
        )
        self.update_sensors()
        if self.check_collision(self.x, self.y):
            self.angle = original_angle
            self.update_sensors()
            self.state_reason = "Cannot rotate due to collision"
            logger.info("Rotation blocked due to collision.")
        else:
            logger.info(f"Rotation successful. New angle: {self.angle:.2f}°")

    def rotate_towards_target(self, target_angle, elapsed_time):
        angle_diff = (target_angle - self.angle + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360
        rotation_speed = CAR_ROTATION_SPEED * elapsed_time
        if abs(angle_diff) > rotation_speed:
            angle_change = rotation_speed if angle_diff > 0 else -rotation_speed
        else:
            angle_change = angle_diff
        self.rotate(angle_change)

    def move_forward(self, elapsed_time):
        distance = self.speed * elapsed_time
        dist_pixels = distance * SCALE
        new_x = self.x + dist_pixels * math.cos(math.radians(self.angle))
        new_y = self.y - dist_pixels * math.sin(math.radians(self.angle))
        if not self.check_collision(new_x, new_y):
            self.x = new_x
            self.y = new_y
            self.update_sensors()
            logger.debug(f"Moved to ({self.x:.2f}, {self.y:.2f})")
        else:
            logger.warning("Collision detected! Movement blocked.")

    def check_point_reached(self):
        if not self.current_target:
            return False
        target_x, target_y = self.current_target
        distance = math.hypot(target_x - self.x, target_y - self.y)

        # --- ADDED DEBUG LOGS ---
        logger.debug(
            f"[check_point_reached] Dist to waypoint: {distance:.2f}, threshold={self.threshold}"
        )

        if distance < self.threshold:
            logger.info(f"Reached point ({target_x}, {target_y})")
            waypoint_name = self.get_waypoint_name(self.current_target)
            if waypoint_name:
                self.current_location_name = waypoint_name
                logger.info(f"Updated current_location_name -> {waypoint_name}")

                # Calculate distance from previous waypoint
                current_waypoint_real = self.waypoint_real_dict.get(waypoint_name)
                if current_waypoint_real and self.previous_waypoint_real:
                    distance_meters = math.hypot(
                        current_waypoint_real[0] - self.previous_waypoint_real[0],
                        current_waypoint_real[1] - self.previous_waypoint_real[1],
                    )
                    logger.info(
                        f"Distance from '{self.get_waypoint_name_from_real(self.previous_waypoint_real)}' to '{waypoint_name}': {distance_meters:.2f} meters"
                    )
                    # Send distance to Arduino
                    # self.send_distance_to_arduino(distance_meters)

                    # Update previous waypoint
                    self.previous_waypoint_real = current_waypoint_real

            if self.path:
                self.current_target = self.path.pop(0)
                next_wp_name = self.get_waypoint_name(self.current_target)
                self.state_reason = f"Moving towards waypoint {next_wp_name}"
                logger.info(f"New target set -> {self.current_target} ({next_wp_name})")
            else:
                self.current_location_name = self.destination_name
                self.current_target = None
                self.moving = False
                if not self.is_returning_to_start:
                    msg = f"Reached {self.destination_name.replace('_', ' ')}. Done or need something else?"
                    self.prompt_queue.put(msg)
                    self.state_reason = "Awaiting user choice"
                    logger.info(f"Prompt enqueued: {msg}")
                else:
                    self.is_returning_to_start = False
                    self.state_reason = "At Start Point"
                    logger.info("Robot has returned to Start Point.")
                return True
        return False

    def get_waypoint_name(self, position):
        for name, pos in zip(self.waypoint_names, self.waypoints):
            if position == pos:
                return name
        return None

    def check_collision(self, new_x, new_y):
        original_x, original_y = self.x, self.y
        self.x, self.y = new_x, new_y

        collision = False
        for wall in self.walls:
            cx = max(wall.rect.left, min(self.x, wall.rect.right))
            cy = max(wall.rect.top, min(self.y, wall.rect.bottom))
            dist = math.hypot(self.x - cx, self.y - cy)
            if dist < ROBOT_VISUAL_DIAMETER / 2:
                logger.warning(f"Collision w/ wall: {wall.rect}")
                collision = True
                break

        if collision:
            self.x, self.y = original_x, original_y
        return collision

    def set_target(self, target_point, destination_name):
        logger.info(f"Setting target to {destination_name} at position {target_point}")
        self.destination_name = destination_name
        self.moving = True
        self.state_reason = f"Moving towards {destination_name.replace('_', ' ')}"
        self.update_sensors()

        path_key = (self.current_location_name.lower(), destination_name.lower())
        if path_key in self.waypoint_paths:
            path_names = self.waypoint_paths[path_key]
            logger.info(f"Path for {path_key}: {path_names}")
            self.path = [self.waypoint_dict[wp] for wp in path_names]
            logger.info(f"Resolved path coords: {self.path}")
        else:
            self.path = []
            logger.warning(f"No predefined path for {path_key}. Direct target set.")

        # Update current_target
        if self.path:
            self.current_target = self.path.pop(0)
            next_wp_name = self.get_waypoint_name(self.current_target)
            self.state_reason = f"Moving towards waypoint {next_wp_name}"
            logger.info(f"New target set -> {self.current_target} ({next_wp_name})")

            # Update current_waypoint_index
            try:
                self.current_waypoint_index = self.waypoint_names.index(next_wp_name)
            except ValueError:
                logger.error(
                    f"Waypoint name {next_wp_name} not found in waypoint_names."
                )
                self.current_waypoint_index = -1  # Invalid index
        else:
            self.current_target = target_point
            logger.info(f"Set current target directly to: {self.current_target}")
            self.current_waypoint_index = -1  # Indicates no further waypoints

        # Calculate total distance
        total_distance = self.calculate_total_distance()

        # Calculate target angle
        target_angle = self.get_target_angle()
        angle_diff = (target_angle - self.angle + 360) % 360

        # Normalize angle_diff to [-180, 180]
        if angle_diff > 180:
            angle_diff -= 360

        # Assign angle_diff
        self.angle_diff = angle_diff
        logger.debug(f"angle_diff set to: {self.angle_diff:.2f} degrees")

        # Set next_waypoint for look-ahead
        self.set_next_waypoint()

        # Determine turn_distance based on next_waypoint
        if self.next_waypoint:
            target_angle_next = self.get_angle_to_next_waypoint()
            angle_diff_next = (target_angle_next - self.angle + 360) % 360
            if angle_diff_next > 180:
                angle_diff_next -= 360
            logger.debug(f"Next angle_diff: {angle_diff_next:.2f} degrees")

            if not math.isclose(angle_diff_next, 0.0, abs_tol=1e-2):
                # Turn is required for the next segment
                # Calculate the distance from current position to the turn waypoint
                turn_wp_coords = self.next_waypoint
                distance_to_turn_wp = math.hypot(
                    turn_wp_coords[0] - self.x, turn_wp_coords[1] - self.y
                )
                # Set turn_distance slightly before the turn waypoint
                # For example, 0.5 meters before
                self.turn_distance = max(
                    0.5, distance_to_turn_wp - 0.5
                )  # Ensure it's non-negative
                logger.info(
                    f"Turn required for next segment. Setting turn_distance to {self.turn_distance:.2f} meters."
                )
            else:
                # No turn required for next segment
                self.turn_distance = 7.0  # Default turn distance
                logger.info(
                    f"No turn required for next segment. Setting turn_distance to {self.turn_distance:.2f} meters."
                )
        else:
            # No next_waypoint; set default turn_distance
            self.turn_distance = 7.0
            logger.info(
                f"No next waypoint. Setting default turn_distance to {self.turn_distance:.2f} meters."
            )

        # Determine if immediate rotation is needed
        if math.isclose(self.angle_diff, 0.0, abs_tol=1e-2):
            self.turn_distance = 0.0  # No turn needed
            self.turn_performed = True  # Mark as performed to skip rotation
            logger.info(
                "No turn needed for angle difference 0°. Proceeding directly to target."
            )
        elif math.isclose(abs(self.angle_diff), 180.0, abs_tol=1e-2):
            self.turn_distance = 0.0  # Immediate turn
            logger.info("180° turn detected. Turn will be performed immediately.")
        else:
            # If turn_distance not set by next_waypoint logic, set to default
            if not hasattr(self, "turn_distance") or self.turn_distance <= 0.0:
                self.turn_distance = 7.0
                logger.info(
                    f"Setting default turn_distance to {self.turn_distance:.2f} meters."
                )

        # Reset turn_performed flag unless angle_diff is 0
        if not math.isclose(angle_diff, 0.0, abs_tol=1e-2):
            self.turn_performed = False
        else:
            self.turn_performed = True  # Already handled

        # Perform immediate turn if needed (only for 180° turns)
        if self.turn_distance == 0.0 and not math.isclose(
            angle_diff, 0.0, abs_tol=1e-2
        ):
            self.rotate(angle_diff)
            self.turn_performed = True
            logger.info("Performed immediate 180° turn.")

        # 1) Build a list that starts with current (x,y) and includes the path
        full_waypoints = []
        full_waypoints.append((self.x, self.y))
        full_waypoints.extend(self.path)

        # 2) Compute single-turn data
        turn_data = self.compute_single_turn(full_waypoints)
        turn_angle = turn_data["turn_angle"]
        turn_dist = turn_data["turn_distance_from_start"]

        # 3) You can optionally log or send them to Arduino
        logger.info(
            f"[SingleTurn] turn_angle={turn_angle:.2f}°, "
            f"turn_distance_from_start={turn_dist:.2f} m"
        )

        # If you want to include them in the same command string:
        command_str = (
            f"DISTANCE {total_distance:.2f} "
            f"ANGLE {angle_diff:.2f} "
            f"TURN_DIST {turn_dist:.2f} "
            f"TURN_ANGLE {turn_angle:.2f}"
        )
        self.send_command(command_str)
        logger.info(
            f"Total distance to {destination_name}: {total_distance:.2f} meters, "
            f"Angle difference: {angle_diff:.2f} degrees, "
            f"First turn angle: {turn_angle:.2f}°, occurs at {turn_dist:.2f} m"
        )

    def update(self):
        current_time = pygame.time.get_ticks()
        elapsed_time = (current_time - self.last_update_time) / 1000.0
        self.last_update_time = current_time

        logger.debug(
            f"[update] Start: moving={self.moving}, current_target={self.current_target}, "
            f"current_location={self.current_location_name}, arduino_obstacle={self.arduino_obstacle_detected}, "
            f"reason={self.state_reason}"
        )

        if self.moving and self.current_target:
            self.started_moving = True

            # Check if Arduino forced us to stop due to external obstacle
            if self.arduino_obstacle_detected:
                self.moving = False
                self.state_reason = "Obstacle detected by Arduino"
                logger.debug("[update] Arduino signaled obstacle; stopping movement.")
                if self.started_moving and not self.obstacle_response_sent:
                    response_queue.put("Excuse me, could you please let me pass?")
                    self.obstacle_response_sent = True
                return

            # Check local sensors for obstacles
            sensor_data = self.check_sensors()
            obstacles = [d for (a, d) in sensor_data if d]
            dist_to_target = math.hypot(
                self.current_target[0] - self.x, self.current_target[1] - self.y
            )

            # If we're close enough to the waypoint, ignore sensor obstacles
            if dist_to_target < self.threshold * 2:
                obstacles = []

            if obstacles:
                # Something is in front of us, so we pause
                self.moving = False
                self.state_reason = "Waiting for obstacle to clear"
                logger.debug("[update] Sensors show obstacle(s); stopping movement.")
                if self.started_moving and not self.obstacle_response_sent:
                    response_queue.put(
                        "Hi! I'm the campus GuideBot. Could you help clear the way?"
                    )
                    self.obstacle_response_sent = True

            else:
                # Clear to move
                self.obstacle_response_sent = False

                # Calculate distance to target in meters
                distance_meters = self.calculate_distance_to_target()

                logger.debug(
                    f"[update] Distance to target: {distance_meters:.2f} meters, "
                    f"Turn distance: {self.turn_distance:.2f} meters, "
                    f"Turn performed: {self.turn_performed}"
                )

                # Determine if a turn is needed based on next_waypoint
                if self.next_waypoint:
                    target_angle_next = self.get_angle_to_next_waypoint()
                    angle_diff_next = (target_angle_next - self.angle + 360) % 360
                    if angle_diff_next > 180:
                        angle_diff_next -= 360
                    logger.debug(f"Next angle_diff: {angle_diff_next:.2f} degrees")

                    # If a turn is needed, set turn_distance accordingly
                    if not math.isclose(angle_diff_next, 0.0, abs_tol=1e-2):
                        # Determine at what distance before current target to perform the turn
                        # For simplicity, set to half of the remaining distance to target
                        self.turn_distance = min(7.0, distance_meters / 2)
                        logger.info(
                            f"Turn required. Setting turn_distance to {self.turn_distance:.2f} meters."
                        )
                    else:
                        # No turn needed for next segment
                        self.turn_distance = 7.0  # Default
                else:
                    # No next waypoint; set default turn_distance
                    self.turn_distance = 7.0

                if distance_meters <= self.turn_distance and not self.turn_performed:
                    # Perform the turn only if angle_diff is not 0
                    if not math.isclose(self.angle_diff, 0.0, abs_tol=1e-2):
                        self.rotate(self.angle_diff)
                        self.turn_performed = True
                        self.state_reason = "Performing turn"
                        logger.info(
                            f"Performing turn of {self.angle_diff:.2f} degrees at {distance_meters:.2f} meters from target."
                        )
                    else:
                        # If angle_diff is 0, mark turn as performed without rotating
                        self.turn_performed = True
                        logger.info(
                            f"No turn needed (angle_diff={self.angle_diff:.2f} degrees). Proceeding to target."
                        )
                else:
                    # Continue moving forward
                    t_angle = self.get_target_angle()
                    a_diff = (t_angle - self.angle + 360) % 360

                    # Shortest angle difference
                    if a_diff > 180:
                        a_diff -= 360

                    rotate_speed = CAR_ROTATION_SPEED * elapsed_time
                    logger.debug(
                        f"[update] No obstacle. TargetAngle={t_angle:.1f}°, "
                        f"CurrentAngle={self.angle:.1f}°, a_diff={a_diff:.1f}°"
                    )

                    # Check if we need to rotate first
                    if abs(a_diff) > rotate_speed:
                        self.rotate_towards_target(t_angle, elapsed_time)
                        self.state_reason = "Rotating towards target"
                        logger.debug("[update] Rotating toward target.")
                    else:
                        # Align exactly, then move forward
                        self.angle = t_angle
                        self.move_forward(elapsed_time)
                        self.state_reason = "Moving forward"
                        logger.debug(
                            f"[update] Moved forward. New pos=({self.x:.2f},{self.y:.2f})"
                        )
                        self.check_point_reached()

    def calculate_distance_to_target(self):
        """
        Calculates the distance to the current target in meters.
        """
        if not self.current_target:
            logger.warning("No current target set.")
            return 0.0

        target_x, target_y = self.current_target
        distance_pixels = math.hypot(target_x - self.x, target_y - self.y)
        distance_meters = distance_pixels / SCALE
        logger.debug(f"Calculated distance to target: {distance_meters:.2f} meters")
        return distance_meters

    def draw_status(self, surface):
        font = pygame.font.SysFont(None, 28)
        status = "MOVING" if self.moving else "STOPPED"
        status_text = font.render(f"Robot Status: {status}", True, BLUE)
        reason_text = font.render(f"Reason: {self.state_reason}", True, BLUE)
        zoom_text = font.render(f"Zoom: {self.camera.zoom:.1f}x", True, RED_COLOR)

        x, y = 20, HEIGHT + 20
        surface.blit(status_text, (x, y))
        surface.blit(reason_text, (x + status_text.get_width() + 40, y))
        surface.blit(
            zoom_text,
            (x + status_text.get_width() + 40 + reason_text.get_width() + 40, y),
        )
        self.draw_zoom_bar(surface)

    def draw_zoom_bar(self, surface):
        bar_width, bar_height = 200, 20
        x, y = 20, HEIGHT + 70
        pygame.draw.rect(surface, BLACK, (x, y, bar_width, bar_height), 2)
        min_zoom, max_zoom = 0.5, 5.0
        z_norm = (self.camera.zoom - min_zoom) / (max_zoom - min_zoom)
        z_norm = max(0.0, min(1.0, z_norm))
        filled_w = int(bar_width * z_norm)
        pygame.draw.rect(
            surface, RED_COLOR, (x + 1, y + 1, filled_w - 2, bar_height - 2)
        )

        font = pygame.font.SysFont(None, 24)
        zoom_text = font.render(f"Zoom: {self.camera.zoom:.1f}x", True, BLACK)
        surface.blit(zoom_text, (x, y - 30))

    def draw(self, surface, camera):
        robot_pos = camera.apply((self.x, self.y))
        pygame.draw.circle(
            surface,
            GREEN if self.moving else RED_COLOR,
            robot_pos,
            ROBOT_VISUAL_DIAMETER // 2,
        )
        end_x = self.x + (ROBOT_VISUAL_DIAMETER / 2) * math.cos(
            math.radians(self.angle)
        )
        end_y = self.y - (ROBOT_VISUAL_DIAMETER / 2) * math.sin(
            math.radians(self.angle)
        )
        end_scr = camera.apply((end_x, end_y))
        pygame.draw.line(surface, BLACK, robot_pos, end_scr, 2)

        sensor_data = self.check_sensors()
        for (angle, (sx, sy)), (_, blocked) in zip(self.sensors, sensor_data):
            c = RED_COLOR if (blocked or self.arduino_obstacle_detected) else GREEN
            s_end = camera.apply((sx, sy))
            pygame.draw.line(surface, c, robot_pos, s_end, 2)
            pygame.draw.circle(surface, c, s_end, 3)

        font = pygame.font.SysFont(None, 24)
        for idx, (wp_x, wp_y) in enumerate(self.waypoints):
            color = (
                GREEN
                if self.waypoint_names[idx] == self.current_location_name
                else BLUE
            )
            wp_scr = camera.apply((wp_x, wp_y))
            pygame.draw.circle(surface, color, wp_scr, 8)
            wname = self.waypoint_names[idx]
            img = font.render(wname, True, BLACK)
            trect = img.get_rect()
            trect.center = (wp_scr[0], wp_scr[1] + 15)
            surface.blit(img, trect)

        self.previous_positions.append(robot_pos)
        if len(self.previous_positions) > 20:
            self.previous_positions.pop(0)
        if len(self.previous_positions) > 1:
            pygame.draw.lines(surface, BLUE, False, self.previous_positions, 2)

    def send_command(self, command):
        if self.serial_reader:
            self.serial_reader.send_command(command)
            logger.info(f"Sent to Arduino: '{command}'")
        else:
            logger.error("SerialReader not assigned to CarRobot.")

    def process_commands(self):
        try:
            while True:
                command = command_queue.get_nowait()
                logger.info(f"Processing command from Flask: {command}")
                if command.startswith("go_to_"):
                    loc = command[len("go_to_") :]
                    loc_norm = loc.replace("-", "_").replace(" ", "_").lower()
                    if (
                        loc_norm.lower() in VALID_DOCTORS
                        or loc_norm in self.waypoint_names
                    ):
                        tpoint = self.waypoint_dict.get(
                            loc_norm.lower(), self.waypoint_dict.get(loc_norm)
                        )
                        if tpoint:
                            self.set_target(tpoint, loc_norm)  # **Correct Reference**
                            logger.info(f"Set target -> {loc_norm}: {tpoint}")
                            self.send_command("START_SERVO")
                        else:
                            logger.warning(f"Unknown location: {loc_norm}")
                elif command == "user_choice_done":
                    logger.info("Received 'user_choice_done' command.")
                    self.return_to_start()  # **Correct Reference**
                    response_queue.put("Goodbye, going to start point.")
                elif command == "user_choice_another":
                    self.state_reason = "Waiting for new command"
                    response_queue.put("How may I help you further?")
        except queue.Empty:
            pass

    def process_responses(self):
        try:
            while True:
                resp = response_queue.get_nowait()
                logger.info(f"Processing response: {resp}")
                threading.Thread(
                    target=self.perform_tts, args=(resp,), daemon=True
                ).start()
        except queue.Empty:
            pass

    def perform_tts(self, text):
        try:
            temp_file = f"response_{uuid.uuid4()}.mp3"
            tts = gTTS(text=text, lang="en")
            tts.save(temp_file)
            logger.info(f"TTS audio saved: {temp_file}")
            snd = AudioSegment.from_mp3(temp_file)
            play(snd)
            os.remove(temp_file)
            logger.info(f"Removed TTS audio file: {temp_file}")
        except Exception as e:
            logger.error(f"Error in perform_tts: {e}")

    def run_flask_app(self):
        url = "http://127.0.0.1:5000/"
        threading.Thread(
            target=open_browser_after_delay, args=(url,), daemon=True
        ).start()
        app.run(debug=False, port=5000, use_reloader=False)

    def send_distance_to_arduino(self, distance_meters):
        """
        Sends the distance to Arduino in meters.
        The distance is formatted as 'DISTANCE 7.00' only when distance_meters is approximately 7 meters.
        """
        # Define tolerance to account for floating-point precision
        tolerance = 0.1  # meters

        if abs(distance_meters - 7.0) <= tolerance:
            distance_message = "DISTANCE 7.00"
            self.send_command(distance_message)
            logger.info(f"Sent to Arduino: '{distance_message}'")
        else:
            logger.debug(
                f"Distance {distance_meters:.2f} not equal to 7.00 meters. No message sent."
            )

    def get_waypoint_name_from_real(self, real_coords):
        """
        Retrieves the waypoint name given its real-world coordinates.
        """
        for name, coords in self.waypoint_real_dict.items():
            if coords == real_coords:
                return name
        return "Unknown"

    def calculate_total_distance(self):
        """
        Calculate the total distance from the current location to the final destination.
        Includes distances between all waypoints in the path and the current target.
        """
        total_distance = 0.0

        # Get current location's real-world coordinates
        current_real = self.waypoint_real_dict.get(self.current_location_name)
        if not current_real:
            logger.error(
                f"Current location '{self.current_location_name}' not found in waypoint_real_dict."
            )
            return 0.0

        # If there's a current target, add distance from current location to current target
        if self.current_target:
            target_name = self.get_waypoint_name(self.current_target)
            target_real = self.waypoint_real_dict.get(target_name)
            if not target_real:
                logger.error(
                    f"Target waypoint '{target_name}' not found in waypoint_real_dict."
                )
                return 0.0
            distance_to_target = math.hypot(
                target_real[0] - current_real[0],
                target_real[1] - current_real[1],
            )
            total_distance += distance_to_target
            logger.info(
                f"Distance from '{self.current_location_name}' to '{target_name}': {distance_to_target:.2f} meters"
            )
            current_real = target_real  # Update current_real to the target's position

        # Add distances between consecutive waypoints in the remaining path
        for wp in self.path:
            wp_name = self.get_waypoint_name(wp)
            wp_real = self.waypoint_real_dict.get(wp_name)
            if not wp_real:
                logger.error(f"Waypoint '{wp_name}' not found in waypoint_real_dict.")
                continue  # Skip if waypoint data is missing
            segment_distance = math.hypot(
                wp_real[0] - current_real[0],
                wp_real[1] - current_real[1],
            )
            total_distance += segment_distance
            logger.info(
                f"Distance from '{self.get_waypoint_name_from_real(current_real)}' to '{wp_name}': {segment_distance:.2f} meters"
            )
            current_real = wp_real  # Update current_real for the next segment

        logger.info(f"Total calculated distance: {total_distance:.2f} meters")
        return total_distance

    def run(self):
        flask_thread = threading.Thread(target=self.run_flask_app, daemon=True)
        flask_thread.start()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    if my < HEIGHT:
                        self.choose_waypoint(mx, my)
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                        if self.camera.zoom < 5.0:
                            self.camera.zoom += 0.5
                            logger.info(f"Zoom -> {self.camera.zoom}")
                            self.update_zoom_related_elements()
                    elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                        if self.camera.zoom > 0.5:
                            self.camera.zoom = max(0.5, self.camera.zoom - 0.5)
                            logger.info(f"Zoom -> {self.camera.zoom}")
                            self.update_zoom_related_elements()

            self.process_commands()
            self.process_responses()

            self.camera.update((self.car.x, self.car.y))
            self.car.update()

            current_moving = self.car.moving
            if current_moving != self.previous_moving_state:
                if current_moving:
                    self.send_command("START_SERVO")
                    logger.info("Send 'START_SERVO' (movement started).")
                else:
                    self.send_command("STOP_SERVO")
                    logger.info("Send 'STOP_SERVO' (movement stopped).")
            self.previous_moving_state = current_moving

            # Clear main simulation area
            sim_rect = pygame.Rect(0, 0, WIDTH, HEIGHT)
            self.screen.fill(WHITE, sim_rect)

            self.draw_walls(self.camera)
            self.car.draw(self.screen, self.camera)
            self.car.draw_status(self.screen)

            # Draw bottom area
            bottom_rect = pygame.Rect(0, HEIGHT, WIDTH, BUTTON_AREA_HEIGHT)
            pygame.draw.rect(self.screen, WHITE, bottom_rect)

            # Ensure status text is on top
            self.car.draw_status(self.screen)

            pygame.display.flip()
            self.clock.tick(FPS)

        self.serial_reader.stop()
        self.serial_reader.join()
        pygame.quit()
        sys.exit()

    def draw_walls(self, camera):
        for wall in self.walls:
            wall.draw(self.screen, camera, color=BLACK)

        # If you have any polygon drawing, keep it here
        # e.g. self.outer_polygon, self.inner_polygon

    def update_zoom_related_elements(self):
        self.car.update_sensors()


def open_browser_after_delay(url, delay=1):
    time.sleep(delay)
    webbrowser.open(url)


class SerialReader(threading.Thread):

    def __init__(self, serial_port: str, baud_rate: int, car_robot: "CarRobot", game):
        super().__init__()
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.running = True
        self.ser = None
        self.car = car_robot
        self.game = game
        self.state = "STOPPED"
        self.lock = Lock()
        self.obstacle_response_sent = False

    def run(self):
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            logger.info(
                f"Connected to Arduino on {self.serial_port} at {self.baud_rate}"
            )
            while self.running:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode("utf-8").strip()
                    if line.startswith("<STATE>") and line.endswith("</STATE>"):
                        st = (
                            line.replace("<STATE>", "")
                            .replace("</STATE>", "")
                            .strip()
                            .upper()
                        )
                        with self.lock:
                            if st == "STOPPED":
                                if self.state != "STOPPED":
                                    logger.info("Arduino -> STOPPED")
                                self.state = "STOPPED"
                                self.car.moving = False
                                self.car.state_reason = "Obstacle detected by Arduino"
                                self.car.arduino_obstacle_detected = True
                                self.game.send_command("STOP_SERVO")
                            elif st == "MOVING":
                                if self.state != "MOVING":
                                    logger.info("Arduino -> MOVING")
                                self.state = "MOVING"
                                self.car.arduino_obstacle_detected = False
                                if self.car.current_target:
                                    self.car.moving = True
                                    self.car.state_reason = "Moving towards waypoint"
                                    self.game.send_command("START_SERVO")
                    elif line.startswith("ACK:"):
                        # Handle acknowledgment messages
                        ack_message = line.replace("ACK:", "").strip()
                        logger.info(f"Arduino Acknowledgment: {ack_message}")
                        # You can add more logic here based on the acknowledgment
        except serial.SerialException as e:
            logger.error(f"Serial Exception: {e}")
            self.running = False

    def send_command(self, command):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(f"{command}\n".encode("utf-8"))
                logger.info(f"Sent command to Arduino: '{command}'")
            except serial.SerialException as e:
                logger.error(f"Failed to send command '{command}': {e}")

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
            logger.info("Serial connection closed.")

    def send_command(self, command):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(f"{command}\n".encode("utf-8"))
                logger.info(f"Sent command to Arduino: {command}")
            except serial.SerialException as e:
                logger.error(f"Failed to send command '{command}': {e}")

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
            logger.info("Serial connection closed.")


class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, TOTAL_HEIGHT))
        pygame.display.set_caption("GuideBot Simulation")
        self.clock = pygame.time.Clock()

        # Initialize Camera with adjusted zoom
        self.camera = Camera(
            WIDTH, HEIGHT, zoom=2.0
        )  # Adjusted initial zoom for better view

        # Create corridor walls based on real-world corridor dimensions
        self.walls = self.create_corridor_walls()

        # Define waypoints in real-world coordinates and scale them
        self.define_waypoints()

        # Instantiate CarRobot with real-world waypoint data
        self.car = CarRobot(
            self.waypoints[0][0],
            self.waypoints[0][1],
            self.waypoints,
            self.waypoint_names,
            self.walls,
            prompt_queue,
            self.camera,  # Pass camera reference
            self.waypoints_real,  # Pass real-world waypoints
            self.waypoint_real_dict,  # Pass real-world waypoint mapping
        )
        # Instantiate SerialReader
        self.serial_reader = SerialReader(SERIAL_PORT, BAUD_RATE, self.car, self)

        # **Assign SerialReader to CarRobot**
        self.car.serial_reader = self.serial_reader  # **Fix Applied Here**

        # Start the SerialReader thread
        self.serial_reader.start()

        self.turn_distance = 0.0  # Initialize turn_distance
        self.turn_performed = False  # Initialize turn_performed flag

        self.previous_moving_state = False

        # For drawing corridor outlines
        self.outer_polygon = None
        self.inner_polygon = None

        # Compute scaled polygons for drawing
        self.compute_corridor_polygons()

    def create_corridor_walls(self):
        """
        Creates a ring-shaped walkable area:
          - Outer boundary: (-2.7, 0) to (42.8, 23.5)
          - Inner boundary: (0, 0) to (39.3, 20)
        Anything outside the outer boundary or inside the inner boundary is a 'wall'.
        """
        walls = []

        # Scaling functions
        def scale_x(rx):
            return PAD + int((rx - (-2.7)) * SCALE)

        def scale_y(ry):
            # Invert y-axis for Pygame
            return PAD + int((real_height - ry) * SCALE)

        # Define outer corridor rectangle with positive height
        outer_rect = pygame.Rect(
            scale_x(-2.7),
            PAD,  # Set y to PAD to ensure positive height
            int((42.8 - (-2.7)) * SCALE),
            int(23.5 * SCALE),  # Positive height
        )

        # Define inner corridor rectangle with positive height
        inner_rect = pygame.Rect(
            scale_x(0),
            scale_y(20),  # Directly set y without subtraction
            int((39.3 - 0) * SCALE),
            int(20 * SCALE),  # Positive height
        )

        # Debugging: Log the rect dimensions
        logger.info(f"Outer Rect: {outer_rect}")
        logger.info(f"Inner Rect: {inner_rect}")

        # 1) Outer walls: Everything outside the outer_rect
        # Split into four rectangles: top, bottom, left, right
        top_wall = Wall(pygame.Rect(0, 0, WIDTH, outer_rect.top))
        bottom_wall = Wall(
            pygame.Rect(0, outer_rect.bottom, WIDTH, HEIGHT - outer_rect.bottom)
        )
        left_wall = Wall(pygame.Rect(0, PAD, outer_rect.left, outer_rect.height))
        right_wall = Wall(
            pygame.Rect(
                outer_rect.right, PAD, WIDTH - outer_rect.right, outer_rect.height
            )
        )

        # 2) Inner walls: The inner_rect itself
        inner_wall = Wall(inner_rect)

        # Add them to the walls list
        walls.extend([top_wall, bottom_wall, left_wall, right_wall, inner_wall])

        # Additional Debugging: Verify all walls have positive dimensions
        for wall in walls:
            rect = wall.rect
            if rect.width <= 0 or rect.height <= 0:
                logger.error(f"Invalid wall dimensions: {rect}")
            else:
                logger.info(f"Valid wall created: {rect}")

        return walls

    def compute_corridor_polygons(self):
        """
        For drawing only: compute lists of (x,y) points in screen coords
        for the outer boundary and inner boundary polygons.
        """
        self.outer_polygon = [
            (scale_x, scale_y)
            for (scale_x, scale_y) in [
                (
                    PAD + int((-2.7 - (-2.7)) * SCALE),
                    PAD + int((real_height - 0) * SCALE),
                ),
                (
                    PAD + int((42.8 - (-2.7)) * SCALE),
                    PAD + int((real_height - 0) * SCALE),
                ),
                (
                    PAD + int((42.8 - (-2.7)) * SCALE),
                    PAD + int((real_height - 23.5) * SCALE),
                ),
                (
                    PAD + int((-2.7 - (-2.7)) * SCALE),
                    PAD + int((real_height - 23.5) * SCALE),
                ),
            ]
        ]

        self.inner_polygon = [
            (scale_x, scale_y)
            for (scale_x, scale_y) in [
                (PAD + int((0 - (-2.7)) * SCALE), PAD + int((real_height - 0) * SCALE)),
                (
                    PAD + int((39.3 - (-2.7)) * SCALE),
                    PAD + int((real_height - 0) * SCALE),
                ),
                (
                    PAD + int((39.3 - (-2.7)) * SCALE),
                    PAD + int((real_height - 20) * SCALE),
                ),
                (
                    PAD + int((0 - (-2.7)) * SCALE),
                    PAD + int((real_height - 20) * SCALE),
                ),
            ]
        ]

    def define_waypoints(self):
        """
        Define waypoints in real-world coordinates, ensuring they lie within the corridor.
        Then scale them to screen coordinates.
        """
        # Define waypoints in real-world coordinates (meters)
        waypoints_real = [
            (2.5, 21.75),  # start
            (9.5, 21.75),  # m415
            (16.5, 21.75),  # m416
            (23.5, 21.75),  # admission
            (30.5, 21.75),  # dr_nada
            (37.5, 21.75),  # dr_omar
            (41.05, 21.75),  # right_corner
            (41.05, 15.75),  # dr_slim
        ]
        self.waypoint_names = [
            "start",
            "m415",
            "m416",
            "admission",
            "dr_nada",
            "dr_omar",
            "right_corner",
            "dr_slim",
        ]

        # Scaling functions
        def scale_x(rx):
            return PAD + int((rx - (-2.7)) * SCALE)

        def scale_y(ry):
            # Invert y-axis for Pygame
            return PAD + int((real_height - ry) * SCALE)

        # Scale waypoints to screen coordinates
        self.waypoints = [(scale_x(rx), scale_y(ry)) for (rx, ry) in waypoints_real]
        self.waypoints_real = waypoints_real  # Store real-world coordinates

        # Create mappings for scaled and real-world coordinates
        self.waypoint_dict = {
            name: position
            for name, position in zip(self.waypoint_names, self.waypoints)
        }
        self.waypoint_real_dict = {
            name: position
            for name, position in zip(self.waypoint_names, self.waypoints_real)
        }

        # Debugging: Log the scaled and real-world waypoints
        for name, pos in self.waypoint_dict.items():
            logger.info(f"Waypoint {name}: {pos} (Scaled)")
        for name, pos in self.waypoint_real_dict.items():
            logger.info(f"Waypoint {name}: {pos} (Real-world)")

    def draw_walls(self, camera):
        for wall in self.walls:
            wall.draw(self.screen, camera, color=BLACK)

        # Draw corridor outlines just for visual reference:
        if self.outer_polygon:
            transformed_outer = [camera.apply(pos) for pos in self.outer_polygon]
            pygame.draw.polygon(self.screen, BLUE, transformed_outer, 3)
        if self.inner_polygon:
            transformed_inner = [camera.apply(pos) for pos in self.inner_polygon]
            pygame.draw.polygon(self.screen, RED_COLOR, transformed_inner, 3)

    def choose_waypoint(self, mouse_x, mouse_y):
        closest_index = None
        min_distance = float("inf")
        for idx, (wp_x, wp_y) in enumerate(self.waypoints):
            # Apply camera to get screen position
            wp_screen_x, wp_screen_y = self.camera.apply((wp_x, wp_y))
            distance = math.hypot(mouse_x - wp_screen_x, mouse_y - wp_screen_y)
            if distance < min_distance and distance < 50:
                closest_index = idx
                min_distance = distance
        if closest_index is not None:
            destination_name = self.waypoint_names[closest_index]
            target_point = self.waypoints[closest_index]
            self.car.set_target(target_point, destination_name)
            logger.info(
                f"Selected waypoint {destination_name}: ({target_point[0]}, {target_point[1]})"
            )
            self.send_command("START_SERVO")

    def send_command(self, command):
        self.serial_reader.send_command(command)

    def process_commands(self):
        try:
            while True:
                command = command_queue.get_nowait()
                logger.info(f"Processing command from Flask: {command}")
                if command.startswith("go_to_"):
                    location = command[len("go_to_") :]
                    location_normalized = (
                        location.replace("-", "_").replace(" ", "_").lower()
                    )
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
                elif command == "user_choice_done":
                    logger.info("Received 'user_choice_done' command.")
                    self.car.return_to_start()
                    response_queue.put("Goodbye, going to start point.")
                elif command == "user_choice_another":
                    self.car.state_reason = "Waiting for new command"
                    response_queue.put("How may I help you further?")
        except queue.Empty:
            pass

    def process_responses(self):
        try:
            while True:
                response = response_queue.get_nowait()
                logger.info(f"Processing response: {response}")
                threading.Thread(
                    target=self.perform_tts, args=(response,), daemon=True
                ).start()
        except queue.Empty:
            pass

    def perform_tts(self, text):
        try:
            temp_file = f"response_{uuid.uuid4()}.mp3"
            tts = gTTS(text=text, lang="en")
            tts.save(temp_file)
            logger.info(f"TTS audio saved as {temp_file}")
            sound = AudioSegment.from_mp3(temp_file)
            play(sound)
            os.remove(temp_file)
            logger.info(f"TTS audio file {temp_file} removed after playback.")
        except Exception as e:
            logger.error(f"Error in perform_tts: {e}")

    def run_flask_app(self):
        url = "http://127.0.0.1:5000/"
        threading.Thread(
            target=open_browser_after_delay, args=(url,), daemon=True
        ).start()
        app.run(debug=False, port=5000, use_reloader=False)

    def run(self):
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
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                        # Zoom in with upper limit
                        if self.camera.zoom < 5.0:
                            self.camera.zoom += 0.5  # Increment zoom
                            logger.info(f"Zoom increased to {self.camera.zoom}")
                            self.update_zoom_related_elements()
                    elif (
                        event.key == pygame.K_MINUS or event.key == pygame.K_UNDERSCORE
                    ):
                        # Zoom out with lower limit
                        if self.camera.zoom > 0.5:
                            self.camera.zoom = max(
                                0.5, self.camera.zoom - 0.5
                            )  # Decrement zoom with a minimum limit
                            logger.info(f"Zoom decreased to {self.camera.zoom}")
                            self.update_zoom_related_elements()

            self.process_commands()
            self.process_responses()

            # Update camera to follow the robot
            self.camera.update((self.car.x, self.car.y))

            self.car.update()
            current_moving_state = self.car.moving
            if current_moving_state != self.previous_moving_state:
                if current_moving_state:
                    self.send_command("START_SERVO")
                    logger.info(
                        "Command 'START_SERVO' sent due to state transition to MOVING."
                    )
                else:
                    self.send_command("STOP_SERVO")
                    logger.info(
                        "Command 'STOP_SERVO' sent due to state transition to STOPPED."
                    )
            self.previous_moving_state = current_moving_state

            # Clear the main simulation area (top 600x450) with white background
            simulation_rect = pygame.Rect(0, 0, WIDTH, HEIGHT)
            self.screen.fill(WHITE, simulation_rect)

            # Draw corridor walls and polygons
            self.draw_walls(self.camera)
            # Draw the robot
            self.car.draw(self.screen, self.camera)
            # Draw status text and zoom indicators in the bottom area
            self.car.draw_status(self.screen)

            # Draw the bottom status area with a white background
            bottom_rect = pygame.Rect(0, HEIGHT, WIDTH, BUTTON_AREA_HEIGHT)
            pygame.draw.rect(self.screen, WHITE, bottom_rect)

            # Ensure status texts are drawn on top of the white background
            self.car.draw_status(self.screen)

            pygame.display.flip()
            self.clock.tick(FPS)

        self.serial_reader.stop()
        self.serial_reader.join()
        pygame.quit()
        sys.exit()

    def draw_grid(self):
        grid_color = DARK_GRAY
        grid_spacing = 50  # pixels

        for x in range(0, WIDTH, grid_spacing):
            pygame.draw.line(self.screen, grid_color, (x, 0), (x, HEIGHT), 1)
        for y in range(0, HEIGHT, grid_spacing):
            pygame.draw.line(self.screen, grid_color, (0, y), (WIDTH, y), 1)

    def update_zoom_related_elements(self):
        """Update elements that depend on zoom level, such as sensors."""
        self.car.update_sensors()


def open_browser_after_delay(url, delay=1):
    time.sleep(delay)
    webbrowser.open(url)


if __name__ == "__main__":
    try:
        game = Game()
        game.run()
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
