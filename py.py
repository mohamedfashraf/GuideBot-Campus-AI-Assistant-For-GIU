# Define a set of valid rooms (can be easily extended)
VALID_ROOMS = {
    "M215",
    "M216",
    "ADMISSION",
    "financial",
    "student_affairs",
}  # Add more rooms as needed

# Generate labels dynamically, including both prefixed and direct room names
labels = [
    "kill",
    "ask_admission_open",
    "confirm_yes",
    "confirm_no",
    "none",
    "hi",  # Greeting
    "hey",  # Greeting
    "hello",  # Greeting
    "financial",  # Room for financial matters
    "student_affairs",  # Room for course-related or student affairs matters
    "admission",  # Room for admission-related matters
]

# Add dynamic command variations for each room in VALID_ROOMS
for room in VALID_ROOMS:
    labels.append(room)  # Standalone room name
    labels.extend(
        [
            f"go_to_{room}",
            f"navigate_to_{room}",
            f"take_me_to_{room}",
            f"go_to_room_{room}",
            f"take_me_to_room_{room}",
        ]
    )

    from datetime import datetime


def check_room_availability(room):
    """Function to check room availability based on the current day and time."""
    # Weekly schedule with opening and closing times for each room
    weekly_schedule = {
        "financial": {
            "saturday": {"opens_at": "09:00", "closes_at": "17:00"},
            "sunday": {"opens_at": "09:00", "closes_at": "17:00"},
            "monday": {"opens_at": "09:00", "closes_at": "17:00"},
            "tuesday": {"opens_at": "09:00", "closes_at": "17:00"},
            "wednesday": {"opens_at": "09:00", "closes_at": "17:00"},
            "thursday": {"opens_at": "09:00", "closes_at": "17:00"},
        },
        "student_affairs": {
            "saturday": {"opens_at": "10:00", "closes_at": "16:00"},
            "sunday": {"opens_at": "10:00", "closes_at": "16:00"},
            "monday": {"opens_at": "10:00", "closes_at": "16:00"},
            "tuesday": {"opens_at": "10:00", "closes_at": "16:00"},
            "wednesday": {"opens_at": "10:00", "closes_at": "16:00"},
            "thursday": {"opens_at": "10:00", "closes_at": "16:00"},
        },
        "admission": {
            "saturday": {"opens_at": "08:00", "closes_at": "15:00"},
            "sunday": {"opens_at": "08:00", "closes_at": "15:00"},
            "monday": {"opens_at": "08:00", "closes_at": "15:00"},
            "tuesday": {"opens_at": "08:00", "closes_at": "15:00"},
            "wednesday": {"opens_at": "08:00", "closes_at": "15:00"},
            "thursday": {"opens_at": "08:00", "closes_at": "15:00"},
        },
    }

    # Get current day and time
    current_day = datetime.now().strftime("%A").lower()
    current_time = datetime.now().strftime("%H:%M")

    # Check if the room has a schedule and if today is in its schedule
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


def open_application(command):
    global pending_action
    response = ""

    # Normalize command to lowercase for consistent handling
    command = command.strip().lower()
    logging.debug(f"Received command: {command}")

    # 1. Check if command directly mentions any valid room name for navigation
    for room in VALID_ROOMS:
        if room.lower() in command:
            availability = check_room_availability(room.lower())
            if availability["is_open"]:
                response = f"The {room} room is open. Taking you there now."
                command_queue.put(f"go_to_{room.lower()}")
                pending_action = None
            else:
                response = (
                    f"The {room} room is currently closed. "
                    f"It opens at {availability['opens_at']} and closes at {availability['closes_at']}."
                )
            response_queue.put(response)
            return jsonify({"response": response})

    # 2. Handle topic-based inquiries with suggestions for relevant rooms
    if "financial" in command or "fee" in command or "payment" in command:
        # Financial-related topics
        availability = check_room_availability("financial")
        if availability["is_open"]:
            response = "For financial matters, you can visit the financial room. Would you like me to guide you there?"
            with pending_action_lock:
                pending_action = "take_to_financial"
        else:
            response = (
                f"The financial room is currently closed. "
                f"It opens at {availability['opens_at']} and closes at {availability['closes_at']}. Please try again later."
            )

    elif "course" in command or "add" in command or "drop" in command:
        # Course-related topics
        availability = check_room_availability("student_affairs")
        if availability["is_open"]:
            response = "For course-related inquiries, you can head to the student affairs room. Would you like me to guide you there?"
            with pending_action_lock:
                pending_action = "take_to_student_affairs"
        else:
            response = (
                f"The student affairs room is currently closed. "
                f"It opens at {availability['opens_at']} and closes at {availability['closes_at']}. Please try again later."
            )

    elif "admission" in command or "apply" in command or "enroll" in command:
        # Admission-related topics
        availability = check_room_availability("admission")
        if availability["is_open"]:
            response = "It sounds like you need assistance with admissions. The admission room is open. Would you like me to guide you there?"
            with pending_action_lock:
                pending_action = "take_to_admission"
        else:
            response = (
                f"The admission room is currently closed. "
                f"It opens at {availability['opens_at']} and closes at {availability['closes_at']}. Please try again later."
            )

    # 3. Handle greeting commands
    elif command in ["hi", "hey", "hello"]:
        response = "Hello there! How can I assist you today?"

    # 4. Handle confirmations if there's a pending action
    elif command == "confirm_yes" and pending_action:
        with pending_action_lock:
            if pending_action == "take_to_financial":
                command_queue.put("go_to_financial")
                response = "Taking you to the financial room."
            elif pending_action == "take_to_student_affairs":
                command_queue.put("go_to_student_affairs")
                response = "Taking you to the student affairs room."
            elif pending_action == "take_to_admission":
                command_queue.put("go_to_admission")
                response = "Taking you to the admission room."
            pending_action = None

    elif command == "confirm_no" and pending_action:
        response = "Okay, let me know if you need anything else."
        pending_action = None

    # 5. Handle 'kill' command
    elif command == "kill":
        response = "Stopping the program. Goodbye!"

    # 6. Handle unrecognized commands
    else:
        response = "I didn't understand the command. Could you please rephrase it?"
        logging.debug(f"Command not recognized: {command}")

    response_queue.put(response)
    return jsonify({"response": response})
