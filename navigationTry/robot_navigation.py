import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import serial
import threading
import queue
import time

# --------------------------- Configuration --------------------------- #
SERIAL_PORT = "COM5"
BAUD_RATE = 9600
SERIAL_TIMEOUT = 1  # seconds
serial_queue = queue.Queue()


# --------------------------- OpenGL Functions --------------------------- #
def draw_point(x, y, z):
    """Draws a point at the given 3D coordinates."""
    glBegin(GL_POINTS)
    glVertex3f(x, y, z)
    glEnd()


def init_opengl():
    """Initialize OpenGL settings."""
    glClearColor(0.0, 0.0, 0.0, 1.0)  # Background color: black
    glEnable(GL_DEPTH_TEST)  # Enable depth testing
    glEnable(GL_POINT_SMOOTH)  # Enable smooth points
    glPointSize(5)  # Size of points

    # Set projection to perspective
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (800 / 600), 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)  # Switch back to modelview


def display():
    """Main OpenGL display loop."""
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Set camera position (z-axis moved farther away to visualize the points)
    gluLookAt(0, 0, 500, 0, 0, 0, 0, 1, 0)  # Camera position

    # Get the latest data from the queue and draw the point
    try:
        while not serial_queue.empty():
            x, y, z = serial_queue.get_nowait()
            print(f"Drawing point at: {x}, {y}, {z}")
            draw_point(x, y, -z)  # Use -z to move points into the scene
    except queue.Empty:
        pass

    pygame.display.flip()


# --------------------------- Serial Reader Thread --------------------------- #
def serial_reader():
    """Reads serial data and adds it to the queue."""
    while True:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8").strip()
            try:
                # The format is "Servo Angle: 127 deg | Distance: 169.52 cm"
                angle_part = line.split("|")[0]  # "Servo Angle: 127 deg"
                distance_part = line.split("|")[1]  # "Distance: 169.52 cm"

                # Extract angle from the angle_part
                angle_str = angle_part.split(":")[1].replace("deg", "").strip()
                angle = float(angle_str)  # Convert angle to float

                # Extract distance from the distance_part
                distance_str = distance_part.split(":")[1].replace("cm", "").strip()
                distance = float(distance_str)  # Convert distance to float

                # Map the angle and distance to 3D coordinates (for now keeping y = 0)
                x = angle
                y = 0
                z = distance

                # Put the values into the queue for rendering
                serial_queue.put((x, y, z))

            except (ValueError, IndexError) as e:
                print(f"Error parsing line: {line} - {e}")
                continue
        time.sleep(0.01)


# --------------------------- Main Loop --------------------------- #
def main_loop():
    """Main loop to run the Pygame OpenGL display."""
    pygame.init()
    display_size = (800, 600)
    pygame.display.set_mode(display_size, DOUBLEBUF | OPENGL)
    init_opengl()

    # Main loop to render 3D points
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        display()


if __name__ == "__main__":
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        threading.Thread(target=serial_reader, daemon=True).start()
        main_loop()
    except serial.SerialException:
        print(f"Failed to connect to {SERIAL_PORT}")
