# GuideBot-Campus AI Assistant for GIU

## Overview

The **GuideBot-Campus AI Assistant** is an innovative AI-powered robotic assistant designed to enhance the campus experience at the German International University (GIU). This project integrates advanced hardware and software to provide interactive navigation, real-time assistance, and conversational AI capabilities for students, staff, and visitors.

## Features

### 1. Voice and Text Command Integration

- Users can interact with the GuideBot using voice or text commands.
- Speech recognition enables hands-free interaction for convenience.

### 2. Natural Language Understanding (NLU)

- The bot’s NLP model interprets user intent, identifying commands such as:
  - Navigating to specific rooms.
  - Addressing financial, course-related, or admission-related inquiries.
- Example: A command related to financial matters directs the user to the financial room after verifying its availability.

### 3. Room and Doctor Availability

- Real-time checking of room and doctor availability.
- Interactive dropdowns allow users to select a doctor and view available dates and times dynamically.

### 4. Conversational Flow

- The bot engages in a conversational manner, confirming room availability and asking for user confirmation before taking action.
- Handles queries like:
  - "Is the admissions office open?"
  - "What time is Dr. Omar available?"

### 5. Interactive Suggestions

- Dynamic suggestions for commands to guide first-time users.
- Prompts for follow-up actions such as "Need something else?" or "I'm done."

### 6. Obstacle Detection and Voice Responses

- Ultrasonic sensors enable the GuideBot to detect obstacles and respond with voice alerts.
- The bot stops and waits if its path is blocked instead of attempting to navigate around the obstacle.

### 7. Hardware Integration

- The GuideBot is equipped with:
  - Ultrasonic sensors for navigation and obstacle detection.
  - Motors for smooth movement across campus.

### 8. Dynamic User Interface

- Developed using modern web technologies for a responsive and engaging experience:
  - Tailwind CSS for styling.
  - JavaScript for dynamic interactivity.
- Features include:
  - Doctor selection dropdowns.
  - Typewriter effects for chatbot responses.
  - Modal-based help guide.

## New Implementations

- Added voice responses when obstacles block the bot’s path.
- Enhanced NLP capabilities for better context understanding.
- Introduced greetings for casual phrases like "Hi" and "Hey."
- Commands for checking room and doctor availability.
- Dynamic suggestions for user-friendly interaction.

## Technical Details

### Frontend

- **Technologies**: HTML, CSS (Tailwind), JavaScript.
- **Features**:
  - Responsive design for mobile and desktop users.
  - Typewriter effects and animated chat bubbles.
  - Dropdown menus for doctor and date selection.

### Backend

- **Technologies**: Flask-based REST API.
- **Capabilities**:
  - Processes user commands and NLP requests.
  - Fetches real-time availability data for rooms and doctors.
  - Handles user interaction prompts.

### Hardware

- **Sensors**: Ultrasonic sensors at the base and head for navigation and obstacle detection.
- **Motors**: Smooth and precise movement across predefined paths.

### Voice Commands

- Integrated speech recognition for real-time voice command execution.
- Commands include:
  - "Take me to Room M415."
  - "I want to apply for the next semester."
  - "Is Dr. Nada available tomorrow?"

## Setup Instructions

### Prerequisites

- Python 3.9 or higher.
- Flask and necessary Python libraries.
- Tailwind CSS (integrated via CDN).
- Modern browser for frontend.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/GuideBot-Campus-AI-Assistant.git
   ```
2. Navigate to the project directory:
   ```bash
   cd GuideBot-Campus-AI-Assistant
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask backend:
   ```bash
   python GuideBot.py
   ```
5. Open the frontend in your browser by launching `index.html`.

## Usage

1. Start the backend server.
2. Open the frontend in a browser.
3. Interact with the GuideBot using voice or text commands:
   - Use the microphone button for voice commands.
   - Type commands into the input box.
4. Follow the bot’s guidance to navigate campus or check availability.

## Future Enhancements

- Implement path planning for better navigation.
- Integrate a larger set of campus facilities and services.
- Add multilingual support for commands and responses.
- Optimize hardware performance for smoother operation.

## Contributors

- **Mohamed Ashraf Metwally** - Project Lead

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Special thanks to the GIU faculty for their support and guidance.
- Inspired by advancements in AI-powered robotics and campus navigation systems.

---

Elevating campus interactions with the power of AI and robotics!
