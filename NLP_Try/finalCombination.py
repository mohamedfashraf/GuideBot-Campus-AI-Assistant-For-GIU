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

# Initialize thread-safe queues for inter-thread communication
command_queue = queue.Queue()
response_queue = queue.Queue()
prompt_queue = queue.Queue()

# -------------------- Constants ---------------------#

pygame.init()

WIDTH, HEIGHT = 800, 600
BUTTON_AREA_HEIGHT = 50
TOTAL_HEIGHT = HEIGHT + BUTTON_AREA_HEIGHT

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

CAR_IMAGE_PATH = "navigationTry/2d-super-car-top-view.png"
CAR_SIZE = (80, 40)
CAR_SPEED = 2
CAR_ROTATION_SPEED = 5
SENSOR_LENGTH = 45
SENSOR_ANGLES = [-30, 0, 30]
WAYPOINT_THRESHOLD = 20
FPS = 60

SERIAL_PORT = "COM5"
BAUD_RATE = 115200

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# -------------------- Flask App Setup ---------------------#

nlp = pipeline(
    "zero-shot-classification",
    model="microsoft/deberta-base-mnli",
    tokenizer="microsoft/deberta-base-mnli",
    framework="pt",
    device=-1,  # for CPU usage -1 for GPU usage 0
)

# Define buildings and rooms
VALID_BUILDINGS = {
    "A": {"name": "Building A", "rooms": {"A101", "A102", "A103"}},
    "M": {
        "name": "Building M",
        "rooms": {"M215", "M216", "M217", "Admission", "Financial", "Student_Affairs"},
    },
    "S": {"name": "Building S", "rooms": {"S301", "S302", "S303"}},
}

# Collect all rooms from buildings
all_rooms = set()
for b_data in VALID_BUILDINGS.values():
    all_rooms.update(b_data["rooms"])

VALID_DOCTORS = {
    "dr_nada",
    "dr_slim",
    "dr_omar",
}

DAYS_OF_WEEK = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]

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

labels.extend(DAYS_OF_WEEK)
labels.append("now")

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


doctor_availability = create_doctor_schedule()


def check_room_availability(room):
    current_day = datetime.now().strftime("%A").lower()
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
    return {"is_open": True}


app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

pending_action = None
pending_action_lock = Lock()
executor = ThreadPoolExecutor(max_workers=4)


@lru_cache(maxsize=128)
def classify_command_cached(command_text):
    result = nlp(
        command_text,
        candidate_labels=tuple(labels),
        hypothesis_template="This text is about {}.",
        multi_label=True,
    )
    confidence_threshold = 0.3
    matched_labels = [
        label
        for label, score in zip(result["labels"], result["scores"])
        if score > confidence_threshold
    ]
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
    ]
    return any(word in response.lower() for word in affirmative_responses)


def is_negative(response):
    negative_responses = ["no", "nope", "not now", "maybe later", "negative"]
    return any(word in response.lower() for word in negative_responses)


def extract_day_from_text(text):
    text = text.lower()
    for day in DAYS_OF_WEEK:
        if day in text:
            return day
    return None


def get_room_opening_times(room, day):
    day = day.lower()
    if room in weekly_schedule and day in weekly_schedule[room]:
        return weekly_schedule[room][day]
    else:
        return None


def get_doctor_schedule(doctor, day):
    day = day.lower()
    doctor = doctor.lower()
    if doctor in doctor_availability and day in doctor_availability[doctor]:
        return doctor_availability[doctor][day]
    else:
        return None


def get_next_opening(room):
    current_day = datetime.now().strftime("%A").lower()
    current_time = datetime.now().strftime("%H:%M")
    days_of_week = DAYS_OF_WEEK
    for i in range(1, 8):
        day_index = (days_of_week.index(current_day) + i) % 7
        next_day = days_of_week[day_index]
        if next_day in weekly_schedule.get(room, {}):
            opening_time = weekly_schedule[room][next_day]["opens_at"]
            return next_day, opening_time
    return None, None


def get_doctor_availability_data(doctor_id):
    current_day = datetime.now().strftime("%A").lower()
    current_time = datetime.now().strftime("%H:%M")
    availability = doctor_availability.get(doctor_id, {})
    if current_day in availability:
        for time_range in availability[current_day]:
            start_time, end_time = map(str.strip, time_range.split("-"))
            if start_time <= current_time <= end_time:
                return {"is_available": True}
        for time_range in availability[current_day]:
            start_time, _ = map(str.strip, time_range.split("-"))
            if current_time < start_time:
                next_availability = f"The next available time is today at {start_time}."
                return {"is_available": False, "next_availability": next_availability}
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
    return render_template("index.html")


@app.route("/doctor_availability", methods=["GET"])
def doctor_availability_endpoint():
    doctor_id = request.args.get("doctor_id")
    if doctor_id:
        availability = doctor_availability.get(doctor_id.lower(), {})
        if availability:
            return jsonify({"status": "success", "data": {doctor_id: availability}})
        return jsonify({"status": "error", "message": "Doctor not found"}), 404
    return jsonify({"status": "success", "data": doctor_availability})


@app.route("/command", methods=["POST"])
def handle_command():
    data = request.json
    logger.debug(f"Received data: {data}")
    command_text = data.get("text", "").strip()
    if command_text:
        logger.debug(f"Command received: {command_text}")
        global pending_action
        if pending_action and (
            pending_action.startswith("go_to_")
            or pending_action == "ask_if_help_needed"
            or pending_action.startswith("check_doctor_availability_")
            or pending_action.startswith("ask_for_day_room_")
            or pending_action.startswith("ask_for_day_doctor_")
        ):
            return open_application(command_text, command_text)
        future = executor.submit(classify_command_cached, command_text)
        predicted_label = future.result()
        logger.info(f"Matched label: {predicted_label}")
        if predicted_label != "none":
            return open_application(predicted_label, command_text)
        else:
            return open_application(command_text, command_text)
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
    if choice == "done":
        command_queue.put("user_choice_done")
        response = "Goodbye, going to start point."
    elif choice == "another":
        command_queue.put("user_choice_another")
        response = "How may I help you further?"
    else:
        response = "Invalid choice."
    return jsonify({"response": response})


@app.route("/post_choice", methods=["POST"])
def post_choice():
    data = request.json
    choice = data.get("choice")
    if choice:
        command_queue.put(choice)
        return jsonify({"status": "success"})
    else:
        return jsonify({"status": "error", "message": "No choice provided."}), 400


def open_browser_after_delay(url, delay=1):
    time.sleep(delay)
    webbrowser.open(url)


def open_application(command, original_command_text):
    global pending_action
    response = ""
    command = command.strip().lower()

    logger.debug(f"open_application called with command: {command}")
    logger.debug(f"Original command text: {original_command_text}")

    # Pending actions that require yes/no
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
            pending_action = None
            logger.debug("User provided a direct request instead of YES/NO.")
            return open_application(command, original_command_text)
        response_queue.put(response)
        logger.info(f"Responding: {response}")
        return jsonify({"response": response})

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

    # No pending action requiring yes/no, proceed normally
    command_normalized = command.replace("dr ", "").replace("doctor ", "").strip()
    availability_query_match = re.search(
        r"(is|are|when|what time)(.*?)(open|close|available)", command_normalized
    )

    if availability_query_match:
        subject = availability_query_match.group(2).strip()
        logger.debug(f"Subject extracted for availability query: {subject}")
        subject_normalized = subject.replace("dr ", "").replace("doctor ", "").strip()
        day_in_text = extract_day_from_text(command_normalized)
        logger.debug(f"Day extracted from query: {day_in_text}")

        if "now" in command_normalized:
            day_in_text = datetime.now().strftime("%A").lower()
            logger.debug(
                f"'Now' detected, setting day_in_text to current day: {day_in_text}"
            )

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
                        opening_times = get_room_opening_times(rm, day_in_text)
                        if opening_times:
                            response = f"The {rm.replace('_', ' ')} opens at {opening_times['opens_at']} and closes at {opening_times['closes_at']} on {day_in_text.capitalize()}."
                        else:
                            response = f"The {rm.replace('_', ' ')} is closed on {day_in_text.capitalize()}."
                        response += " Is there anything else I can assist you with?"
                        pending_action = "ask_if_help_needed"
                        response_queue.put(response)
                        logger.info(f"Responding: {response}")
                        return jsonify({"response": response})
                    else:
                        response = "Do you need the opening times for a specific day?"
                        with pending_action_lock:
                            pending_action = f"ask_for_day_room_{rm}"
                        response_queue.put(response)
                        logger.info(f"Responding: {response}")
                        return jsonify({"response": response})
            if found_room:
                break

        if not found_room:
            # Check if subject is a doctor
            for doc in VALID_DOCTORS:
                doctor_lower = doc.replace("_", " ").lower()
                doc_norm = (
                    doctor_lower.replace("dr ", "").replace("doctor ", "").strip()
                )
                if doc_norm == subject_normalized or doc_norm in subject_normalized:
                    if day_in_text:
                        schedule = get_doctor_schedule(doc, day_in_text)
                        if schedule:
                            schedule_str = ", ".join(schedule)
                            response = f"{doc.replace('_', ' ').title()} is available at the following times on {day_in_text.capitalize()}: {schedule_str}."
                        else:
                            response = f"{doc.replace('_', ' ').title()} is not available on {day_in_text.capitalize()}."
                        response += " Is there anything else I can assist you with?"
                        pending_action = "ask_if_help_needed"
                        response_queue.put(response)
                        logger.info(f"Responding: {response}")
                        return jsonify({"response": response})
                    else:
                        if not day_in_text:
                            day_in_text = datetime.now().strftime("%A").lower()
                        schedule = get_doctor_schedule(doc, day_in_text)
                        if schedule:
                            schedule_str = ", ".join(schedule)
                            response = f"{doc.replace('_', ' ').title()} is available at the following times today: {schedule_str}."
                        else:
                            response = f"{doc.replace('_', ' ').title()} is not available today."
                        response += " Is there anything else I can assist you with?"
                        pending_action = "ask_if_help_needed"
                        response_queue.put(response)
                        logger.info(f"Responding: {response}")
                        return jsonify({"response": response})

    # Identify rooms or doctors normally
    room = None
    doctor = None

    # Check for room
    for b_data in VALID_BUILDINGS.values():
        for rm in b_data["rooms"]:
            if rm.replace("_", " ").lower() in command:
                room = rm
                logger.debug(f"Identified room: {room}")
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
                logger.debug(f"Identified doctor: {doctor}")
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
            keyword in command for keyword in ["giu", "german international university"]
        ):
            response = "Welcome to the German International University! How can I assist you today?"
            response_queue.put(response)
            logger.info(f"Responding: {response}")
            return jsonify({"response": response})

    # Provide response based on identified room or doctor
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
            response = (
                f"{room.replace('_', ' ')} is currently closed and will open on "
                f"{next_open_day.capitalize()} at {next_open_time}. Would you like help with something else?"
            )
            with pending_action_lock:
                pending_action = "ask_if_help_needed"
            response_queue.put(response)
            logger.info(f"Responding: {response}")
        return jsonify({"response": response})

    elif doctor:
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


class Wall:
    def __init__(self, rect):
        self.rect = rect
        self.mask = self.create_mask()

    def create_mask(self):
        wall_surface = pygame.Surface(
            (self.rect.width, self.rect.height), pygame.SRCALPHA
        )
        wall_surface.fill(BLACK)
        return pygame.mask.from_surface(wall_surface)

    def draw(self, surface):
        pygame.draw.rect(surface, BLACK, self.rect)


class CarRobot:
    def __init__(self, x, y, waypoints, waypoint_names, walls, prompt_queue):
        self.start_x = x
        self.start_y = y
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = CAR_SPEED
        self.waypoints = waypoints
        self.waypoint_names = waypoint_names
        self.current_target = None
        self.current_location_name = "Start"
        self.destination_name = None
        self.moving = False
        self.threshold = WAYPOINT_THRESHOLD
        self.walls = walls
        self.load_image()
        self.state_reason = "Waiting for waypoint"
        self.is_returning_to_start = False
        self.prompt_queue = prompt_queue
        self.sensors = []
        self.create_sensors()
        self.path = []
        self.arduino_obstacle_detected = False
        self.obstacle_response_sent = False
        self.started_moving = False
        self.waypoint_dict = {
            name: position
            for name, position in zip(self.waypoint_names, self.waypoints)
        }
        logger.info(f"Waypoint Dictionary Initialized: {self.waypoint_dict}")

    def load_image(self):
        try:
            self.original_image = pygame.image.load(CAR_IMAGE_PATH)
            self.original_image = pygame.transform.scale(self.original_image, CAR_SIZE)
            logger.debug("Car image loaded successfully.")
        except pygame.error as e:
            logger.error(f"Failed to load car image: {e}")
            sys.exit()

    def create_sensors(self):
        self.sensors = []
        for angle_offset in SENSOR_ANGLES:
            sensor_angle = (self.angle + angle_offset) % 360
            sensor_end_x = self.x + SENSOR_LENGTH * math.cos(math.radians(sensor_angle))
            sensor_end_y = self.y - SENSOR_LENGTH * math.sin(math.radians(sensor_angle))
            self.sensors.append((sensor_angle, (sensor_end_x, sensor_end_y)))

    def update_sensors(self):
        self.create_sensors()

    def check_sensors(self):
        sensor_data = []
        for sensor_angle, (sensor_end_x, sensor_end_y) in self.sensors:
            sensor_line = ((self.x, self.y), (sensor_end_x, sensor_end_y))
            obstacle_detected = False
            for wall in self.walls:
                if self.line_rect_intersect(sensor_line, wall.rect):
                    obstacle_detected = True
                    break
            sensor_data.append((sensor_angle, obstacle_detected))
        return sensor_data

    def return_to_start(self):
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
                self.is_returning_to_start = True
                self.state_reason = "Returning to start via checkpoints"
                logger.info(
                    f"Returning to start from {self.current_location_name} following checkpoints."
                )
            else:
                logger.warning(
                    f"No return path defined from {self.current_location_name} to Start."
                )

    def line_rect_intersect(self, line, rect):
        rect_lines = [
            ((rect.left, rect.top), (rect.right, rect.top)),
            ((rect.right, rect.top), (rect.right, rect.bottom)),
            ((rect.right, rect.bottom), (rect.left, rect.bottom)),
            ((rect.left, rect.bottom), (rect.left, rect.top)),
        ]
        for rect_line in rect_lines:
            if self.line_line_intersect(line, rect_line):
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

    def get_target_angle(self):
        if not self.current_target:
            return self.angle
        target_x, target_y = self.current_target
        dx = target_x - self.x
        dy = self.y - target_y
        angle = math.degrees(math.atan2(dy, dx))
        return angle % 360

    def rotate(self, angle_change):
        original_angle = self.angle
        self.angle = (self.angle + angle_change) % 360
        self.update_sensors()
        if self.check_collision(self.x, self.y):
            self.angle = original_angle
            self.update_sensors()
            self.state_reason = "Cannot rotate due to collision"
            self.moving = False

    def rotate_towards_target(self, target_angle):
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
        new_x = self.x + self.speed * math.cos(math.radians(self.angle))
        new_y = self.y - self.speed * math.sin(math.radians(self.angle))
        if not self.check_collision(new_x, new_y):
            self.x = new_x
            self.y = new_y
            self.update_sensors()
        else:
            logger.warning("Collision detected! Movement blocked.")

    def check_point_reached(self):
        if not self.current_target:
            return False
        target_x, target_y = self.current_target
        distance = math.hypot(target_x - self.x, target_y - self.y)
        if distance < self.threshold:
            logger.info(f"Reached point ({target_x}, {target_y})")
            waypoint_name = self.get_waypoint_name(self.current_target)
            if waypoint_name:
                self.current_location_name = waypoint_name
            if self.path:
                self.current_target = self.path.pop(0)
                self.state_reason = f"Moving towards waypoint ({self.current_target[0]}, {self.current_target[1]})"
            else:
                self.current_location_name = self.destination_name
                self.current_target = None
                self.moving = False
                destination_name = self.current_location_name
                if not self.is_returning_to_start:
                    prompt_message = f"Reached {destination_name.replace('_', ' ')}. Are you done or do you need something else?"
                    self.prompt_queue.put(prompt_message)
                    self.state_reason = "Awaiting user choice"
                else:
                    self.is_returning_to_start = False
                    self.state_reason = "At Start Point"
                return True
        return False

    def get_waypoint_name(self, position):
        for name, pos in zip(self.waypoint_names, self.waypoints):
            if position == pos:
                return name
        return None

    def update_mask(self):
        rotated_image = pygame.transform.rotate(self.original_image, self.angle)
        rotated_rect = rotated_image.get_rect(center=(self.x, self.y))
        self.car_mask = pygame.mask.from_surface(rotated_image)
        return rotated_image, rotated_rect

    def check_collision(self, new_x, new_y):
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
        if self.moving and self.current_target:
            self.started_moving = True
            if self.arduino_obstacle_detected:
                self.moving = False
                self.state_reason = "Obstacle detected by Arduino"
                if self.started_moving and not self.obstacle_response_sent:
                    response_queue.put("Excuse me, could you please let me pass?")
                    self.obstacle_response_sent = True
                return
            sensor_data = self.check_sensors()
            obstacles = [detected for angle, detected in sensor_data if detected]
            target_distance = math.hypot(
                self.current_target[0] - self.x, self.current_target[1] - self.y
            )
            if target_distance < self.threshold * 2:
                obstacles = []
            if obstacles:
                self.moving = False
                self.state_reason = "Waiting for obstacle to clear"
                if self.started_moving and not self.obstacle_response_sent:
                    response_queue.put(
                        "Hi! I’m the campus GuideBot. Could you help clear the way?"
                    )
                    self.obstacle_response_sent = True
            else:
                self.obstacle_response_sent = False
                target_angle = self.get_target_angle()
                angle_diff = (target_angle - self.angle + 360) % 360
                if angle_diff > 180:
                    angle_diff -= 360
                if abs(angle_diff) > CAR_ROTATION_SPEED:
                    self.rotate_towards_target(target_angle)
                    self.state_reason = "Rotating towards target"
                else:
                    self.move_forward()
                    self.check_point_reached()
        else:
            if self.arduino_obstacle_detected:
                self.state_reason = "Obstacle detected by Arduino"
                if self.started_moving and not self.obstacle_response_sent:
                    response_queue.put("Excuse me, could you please let me pass?")
                    self.obstacle_response_sent = True
            else:
                self.state_reason = "Stopped"
                self.obstacle_response_sent = False

    def draw_status(self, surface):
        font = pygame.font.SysFont(None, 24)
        status = "MOVING" if self.moving else "STOPPED"
        status_text = font.render(f"Robot Status: {status}", True, BLACK)
        reason_text = font.render(f"Reason: {self.state_reason}", True, BLACK)
        status_width = status_text.get_width()
        margin = 20
        surface.blit(status_text, (10, HEIGHT + 10))
        surface.blit(reason_text, (10 + status_width + margin, HEIGHT + 10))

    def draw(self, surface):
        rotated_image, rect = self.update_mask()
        surface.blit(rotated_image, rect.topleft)
        sensor_data = self.check_sensors()
        for (sensor_angle, (sensor_end_x, sensor_end_y)), (
            angle_check,
            obstacle_detected,
        ) in zip(self.sensors, sensor_data):
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

    waypoint_paths = {
        ("Start", "M215"): ["M215"],
        ("M215", "M216"): ["M216"],
        ("M216", "Admission"): ["Admission"],
        ("Admission", "Start"): ["M216", "M215", "Start"],
        ("Start", "M216"): ["M215", "M216"],
        ("Start", "Admission"): ["M215", "M216", "Admission"],
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
    }


class SerialReader(threading.Thread):
    def __init__(self, serial_port, baud_rate, car, game):
        super().__init__()
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.running = True
        self.ser = None
        self.car = car
        self.game = game
        self.state = "STOPPED"
        self.lock = threading.Lock()
        self.obstacle_response_sent = False

    def run(self):
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
                                self.car.moving = False
                                self.car.state_reason = "Obstacle detected by Arduino"
                                self.car.arduino_obstacle_detected = True
                                self.game.send_command("STOP_SERVO")
                            elif state == "MOVING":
                                if self.state != "MOVING":
                                    logger.info("Arduino: MOVING")
                                self.state = "MOVING"
                                self.car.arduino_obstacle_detected = False
                                if self.car.current_target:
                                    self.car.moving = True
                                    self.car.state_reason = "Moving towards waypoint"
                                    self.game.send_command("START_SERVO")
        except serial.SerialException as e:
            logger.error(f"Serial Exception: {e}")
            self.running = False

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
        self.walls = self.create_walls()
        self.define_waypoints()
        self.car = CarRobot(
            self.waypoints[0][0],
            self.waypoints[0][1],
            self.waypoints,
            self.waypoint_names,
            self.walls,
            prompt_queue,
        )
        self.car.waypoint_dict = self.waypoint_dict
        self.serial_reader = SerialReader(SERIAL_PORT, BAUD_RATE, self.car, self)
        self.serial_reader.start()
        self.previous_moving_state = False

    def create_walls(self):
        wall_rects = [
            pygame.Rect(50, 50, 700, 50),
            pygame.Rect(50, 500, 700, 50),
            pygame.Rect(50, 50, 50, 500),
            pygame.Rect(720, 50, 50, 500),
            pygame.Rect(350, 250, 100, 100),
            pygame.Rect(50, 300, 300, 10),
        ]
        return [Wall(rect) for rect in wall_rects]

    def define_waypoints(self):
        self.waypoints = [
            (150, 150),
            (600, 150),
            (600, 450),
            (150, 450),
            (500, 300),
        ]
        self.waypoint_names = ["Start", "M215", "M216", "Admission", "dr_nada"]
        self.waypoint_dict = {
            name: position
            for name, position in zip(self.waypoint_names, self.waypoints)
        }

    def draw_walls(self):
        for wall in self.walls:
            wall.draw(self.screen)

    def choose_waypoint(self, mouse_x, mouse_y):
        closest_index = None
        min_distance = float("inf")
        for idx, (wp_x, wp_y) in enumerate(self.waypoints):
            distance = math.hypot(mouse_x - wp_x, mouse_y - wp_y)
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
                        location.replace("-", "_").replace(" ", "_").title()
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
                    else:
                        logger.warning(f"Unknown location: {location_normalized}")
                elif command == "user_choice_done":
                    self.car.return_to_start()
                    response_queue.put("Goodbye, going to start point.")
                elif command == "user_choice_another":
                    self.car.state_reason = "Waiting for new command"
                    response_queue.put("How may I help you further")
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
            self.process_commands()
            self.process_responses()
            self.car.update()
            current_moving_state = self.car.moving
            if current_moving_state != self.previous_moving_state:
                if current_moving_state:
                    self.send_command("START_SERVO")
                    logger.debug(
                        "Command 'START_SERVO' sent due to state transition to MOVING."
                    )
                else:
                    self.send_command("STOP_SERVO")
                    logger.debug(
                        "Command 'STOP_SERVO' sent due to state transition to STOPPED."
                    )
            self.previous_moving_state = current_moving_state
            self.screen.fill(WHITE)
            self.draw_walls()
            self.car.draw(self.screen)
            self.car.draw_status(self.screen)
            pygame.display.flip()
            self.clock.tick(FPS)
        self.serial_reader.stop()
        self.serial_reader.join()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    try:
        game = Game()
        game.run()
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
