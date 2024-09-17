import speech_recognition as sr
from gtts import gTTS
import os
import subprocess
from transformers import pipeline

# Initialize the zero-shot classification pipeline
nlp = pipeline("zero-shot-classification")

# Define possible labels or categories
labels = ["open_browser", "open_notepad", "play_music", "turn_on_lights"]


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
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError:
        print("Sorry, there was an issue with the request.")
        return None


def text_to_speech(response):
    tts = gTTS(text=response, lang="en")
    tts.save("response.mp3")
    os.system(
        "start response.mp3"
    )  # On Windows; use "afplay" on macOS or "mpg321" on Linux


def open_application(command):
    if command == "open_browser":
        subprocess.Popen(["start", "chrome"], shell=True)  # Opens Google Chrome
        response = "Opening the browser."
    elif command == "open_notepad":
        subprocess.Popen(["notepad.exe"])  # Opens Notepad
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
        exit()  # Exit the script
    else:
        response = "I didn't understand the command."

    text_to_speech(response)


def main():
    last_command = None

    while True:
        command_text = recognize_speech()

        # Check if the command is "I am done"
        if command_text and "i am done" in command_text:
            text_to_speech("Goodbye!")
            break

        if command_text:
            result = nlp(command_text, candidate_labels=labels)
            predicted_label = result["labels"][0]

            # Avoid repeating the same command
            if predicted_label != last_command:
                open_application(predicted_label)
                last_command = predicted_label
            else:
                print("Ignoring duplicate command:", predicted_label)
                text_to_speech("Ignoring duplicate command.")


if __name__ == "__main__":
    main()
