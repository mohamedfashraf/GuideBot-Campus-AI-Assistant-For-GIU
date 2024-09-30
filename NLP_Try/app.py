from flask import Flask, jsonify, request
from flask_cors import CORS  # Import CORS
from gtts import gTTS
import os
import subprocess
from transformers import pipeline
from pydub import AudioSegment
from pydub.playback import play
import uuid

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the zero-shot classification pipeline with explicit model and clean_up_tokenization_spaces
nlp = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    clean_up_tokenization_spaces=True,
)

# Possible labels to classify commands
labels = [
    "open_browser",
    "open_notepad",
    "play_music",
    "turn_on_lights",
    "kill",
    "none",
]


def text_to_speech(response):
    """Converts text response to speech and plays it."""
    try:
        temp_file = f"response_{uuid.uuid4()}.mp3"
        tts = gTTS(text=response, lang="en")
        tts.save(temp_file)

        # Play the generated speech audio
        sound = AudioSegment.from_mp3(temp_file)
        play(sound)

        # Remove the audio file after playback
        os.remove(temp_file)
    except Exception as e:
        print(f"Error in text_to_speech: {e}")


def open_application(command):
    """Executes the appropriate application based on the command."""
    response = ""
    if command == "open_browser":
        print("Attempting to open the browser...")  # Debugging
        subprocess.Popen(["start", "chrome"], shell=True)
        response = "Opening the browser."
    elif command == "open_notepad":
        print("Attempting to open Notepad...")  # Debugging
        subprocess.Popen(["notepad.exe"])
        response = "Opening Notepad."
    elif command == "play_music":
        response = "Playing music."
        # You can add functionality here to play music
    elif command == "turn_on_lights":
        response = "Turning on the lights."
        # Add smart light control code here if you want
    elif command == "kill":
        response = "Stopping the program. Goodbye!"
        text_to_speech(response)
        # Stop the Flask server or exit (optional)
        return jsonify({"response": response})
    elif command == "none":
        response = "I didn't understand the command."
    else:
        response = "I didn't understand the command."

    text_to_speech(response)
    return jsonify({"response": response})


@app.route("/command", methods=["POST"])
def handle_command():
    """Handles the POST request from the frontend, processes the command, and sends back a response."""
    data = request.json
    print("Received data:", data)  # Debugging statement
    command_text = data.get("text", "")

    if command_text:
        print("Command received:", command_text)  # Debugging statement
        result = nlp(command_text, candidate_labels=labels, multi_label=False)
        print("Model result:", result)  # Debugging statement
        predicted_label = result["labels"][0]
        confidence = result["scores"][0]

        print(
            f"Predicted label: {predicted_label} with confidence {confidence}"
        )  # Debugging

        # If confidence is above 0.5, execute the application
        if confidence > 0.5:
            return open_application(predicted_label)
        else:
            return jsonify(
                {"response": "I'm not sure what you meant. Can you try again?"}
            )

    return jsonify({"response": "No command received."})


if __name__ == "__main__":
    # Running Flask server on port 5000
    app.run(debug=True, port=5000)
