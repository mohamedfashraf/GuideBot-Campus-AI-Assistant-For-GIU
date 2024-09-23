import speech_recognition as sr
from gtts import gTTS
import os
import subprocess
from transformers import pipeline
from pydub import AudioSegment
from pydub.playback import play
import threading
import time
import uuid

# Initialize the zero-shot classification pipeline
nlp = pipeline("zero-shot-classification")

# Define possible labels or categories
labels = [
    "open_browser",
    "open_notepad",
    "play_music",
    "turn_on_lights",
    "kill",
    "none",
]

# Variable to control the assistant's listening state
listening = False

# Global timer
timer = None

# Lock for thread-safe text-to-speech
tts_lock = threading.Lock()

# Flag to indicate if the assistant is currently speaking
is_speaking = False


# Function for text-to-speech
def text_to_speech(response):
    global is_speaking
    with tts_lock:
        if is_speaking:
            print("Already speaking, skipping this response.")
            return  # Skip the speech if already speaking

        is_speaking = True
        try:
            # Generate a unique filename
            temp_file = f"response_{uuid.uuid4()}.mp3"
            tts = gTTS(text=response, lang="en")
            tts.save(temp_file)

            # Play the sound using pydub
            sound = AudioSegment.from_mp3(temp_file)
            play(sound)

            # Attempt to remove the audio file after playback
            try:
                os.remove(temp_file)
            except FileNotFoundError:
                print(f"File {temp_file} not found.")
            except PermissionError:
                print(f"Error removing {temp_file}: File is still in use.")
        except Exception as e:
            print(f"Error in text_to_speech: {e}")
        finally:
            is_speaking = False  # Reset the flag when done speaking


# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for a command...")
        try:
            # Listen with a timeout to prevent indefinite blocking
            audio = recognizer.listen(source, timeout=15)
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase to start")
            return None
    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text.lower()  # Convert text to lowercase
    except sr.UnknownValueError:
        text_to_speech("I did not get it, can you say it again?")
        return None
    except sr.RequestError:
        text_to_speech("Sorry, there was an issue with the request.")
        return None


# Function to open applications
def open_application(command):
    if command == "open_browser":
        subprocess.Popen(["start", "chrome"], shell=True)
        response = "Opening the browser."
    elif command == "open_notepad":
        subprocess.Popen(["notepad.exe"])
        response = "Opening Notepad."
    elif command == "play_music":
        response = "Playing music."
        # Add music-playing code here if needed
    elif command == "turn_on_lights":
        response = "Turning on the lights."
        # Add code to control smart lights if needed
    elif command == "kill":
        response = "Stopping the program. Goodbye!"
        text_to_speech(response)
        if timer is not None:
            timer.cancel()
        exit()
    elif command == "none":
        response = "I didn't understand the command."
    else:
        response = "I didn't understand the command."

    text_to_speech(response)


# Function to handle inactivity timeout
def inactivity_timeout():
    global listening
    if listening:
        print("Inactivity timeout reached.")
        text_to_speech("Goodbye!")
        listening = False


# Function to reset the inactivity timer
def reset_timer():
    global timer
    if timer is not None:
        timer.cancel()
    timer = threading.Timer(15, inactivity_timeout) #! timeout to 15 seconds
    timer.start()
    print("Inactivity timer reset.")


def main():
    global listening, timer
    last_command = None

    while True:
        input("Press Enter to wake up the assistant...")
        listening = True
        text_to_speech("How can I assist you?")
        print("Assistant is active and listening for commands.")

        # Start the inactivity timer
        reset_timer()

        while listening:
            command_text = recognize_speech()

            if command_text:
                # Reset the timer on each recognized command
                reset_timer()

                if "i am done" in command_text:
                    text_to_speech("Goodbye!")
                    listening = False
                    print("Assistant is now inactive.")
                    # Cancel the timer to prevent it from triggering again
                    if timer is not None:
                        timer.cancel()
                    continue

                result = nlp(command_text, candidate_labels=labels, multi_label=False)
                predicted_label = result["labels"][0]
                confidence = result["scores"][0]

                print(
                    f"Predicted label: {predicted_label} with confidence {confidence}"
                )

                # Use a confidence threshold to avoid misclassifications
                if confidence > 0.5:
                    if predicted_label != last_command:
                        open_application(predicted_label)
                        last_command = predicted_label
                    else:
                        print("Ignoring duplicate command:", predicted_label)
                        text_to_speech("Ignoring duplicate command.")
                else:
                    text_to_speech("I'm not sure what you meant. Can you try again?")

        # After listening ends, inform the user
        text_to_speech("The assistant is now inactive. Press Enter to wake up.")


if __name__ == "__main__":
    main()
