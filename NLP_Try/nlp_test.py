import speech_recognition as sr
from gtts import gTTS
import os
import subprocess
from transformers import pipeline
from pydub import AudioSegment
from pydub.playback import play
import threading
import time

# Initialize the zero-shot classification pipeline
nlp = pipeline("zero-shot-classification")

# Define possible labels or categories
labels = ["open_browser", "open_notepad", "play_music", "turn_on_lights"]

# Variable to control the assistant's listening state
listening = False


# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for a command...")
        audio = recognizer.listen(source)
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


# Function for text-to-speech
def text_to_speech(response):
    tts = gTTS(text=response, lang="en")
    temp_file = "response.mp3"
    tts.save(temp_file)

    # Play the sound using pydub
    sound = AudioSegment.from_mp3(temp_file)
    play(sound)

    # Attempt to remove the audio file after playback
    try:
        os.remove(temp_file)
    except PermissionError:
        print("Error removing response.mp3: File is still in use.")


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
    elif command == "turn_on_lights":
        response = "Turning on the lights."
    elif command == "kill":
        response = "Stopping the program. Goodbye!"
        text_to_speech(response)
        exit()
    else:
        response = "I didn't understand the command."

    text_to_speech(response)


# Function to monitor inactivity
def inactivity_timer():
    global listening
    while True:
        time.sleep(30)
        if listening:
            text_to_speech("Goodbye!")
            listening = False  # Stop listening for commands


def main():
    global listening
    last_command = None

    while True:
        input("Press Enter to wake up the assistant...")
        listening = True
        text_to_speech("How can I assist you?")

        # Start the inactivity timer
        timer_thread = threading.Thread(target=inactivity_timer)
        timer_thread.daemon = True
        timer_thread.start()

        while listening:
            command_text = recognize_speech()

            if command_text:
                if "i am done" in command_text:
                    text_to_speech("Goodbye!")
                    listening = False
                    continue

                result = nlp(command_text, candidate_labels=labels)
                predicted_label = result["labels"][0]

                # Avoid repeating the same command
                if predicted_label != last_command:
                    open_application(predicted_label)
                    last_command = predicted_label
                else:
                    print("Ignoring duplicate command:", predicted_label)
                    text_to_speech("Ignoring duplicate command.")

        # After listening ends, inform the user
        text_to_speech("The assistant is now inactive. Press Enter to wake up.")


if __name__ == "__main__":
    main()

