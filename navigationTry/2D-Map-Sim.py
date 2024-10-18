import pygame
import math
import sys
import serial
import threading
import logging
from datetime import datetime

# -------------------- Constants ---------------------#

# Pygame Initialization
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
BUTTON_AREA_HEIGHT = 200  # Increased space for the buttons and prompt
TOTAL_HEIGHT = HEIGHT + BUTTON_AREA_HEIGHT  # Total screen height with buttons area

# Color definitions
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
DARK_GRAY = (169, 169, 169)
LIGHT_GRAY = (211, 211, 211)

# Car properties
CAR_IMAGE_PATH = "navigationTry/2d-super-car-top-view.png"
CAR_SIZE = (100, 50)
CAR_SPEED = 2
CAR_ROTATION_SPEED = 5

# Sensor properties
NUM_SENSORS = 3
SENSOR_LENGTH = 150
SENSOR_FOV = 30  # Field of View in degrees

# Waypoint properties
WAYPOINT_THRESHOLD = 20  # Distance to consider waypoint as reached

# Frame rate
FPS = 60

# Serial communication settings
SERIAL_PORT = "COM5"
BAUD_RATE = 9600

# -------------------- Logging Configuration ---------------------#

# Configure the logging module
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to console
    ],
)

# Create a logger object
logger = logging.getLogger(__name__)

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

    def __init__(self, x, y, waypoints, waypoint_names, walls):
        """
        Initialize the CarRobot.

        Args:
            x (float): Initial x-coordinate.
            y (float): Initial y-coordinate.
            waypoints (list): List of waypoint tuples.
            waypoint_names (list): List of names for each waypoint.
            walls (list): List of Wall objects for collision detection.
        """
        self.start_x = x
        self.start_y = y
        self.x = x
        self.y = y
        self.angle = 0  # Facing right initially
        self.speed = CAR_SPEED
        self.waypoints = waypoints
        self.waypoint_names = waypoint_names
        self.current_waypoint_index = None  # No waypoint selected initially
        self.moving = False  # Car should start stationary
        self.threshold = WAYPOINT_THRESHOLD
        self.walls = walls
        self.sensors = []
        self.num_sensors = NUM_SENSORS
        self.sensor_length = SENSOR_LENGTH
        self.sensor_fov = SENSOR_FOV
        self.load_image()
        self.create_sensors()
        self.obstacle_detected = False  # Obstacle detection status
        self.state_reason = "Waiting for waypoint"  # Reason for stop or move
        self.awaiting_choice = (
            False  # Waiting for user input to continue or go to start
        )
        self.is_returning_to_start = False  # Flag to track returning to start

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
        half_fov = self.sensor_fov / 2
        angle_gap = (
            self.sensor_fov / (self.num_sensors - 1) if self.num_sensors > 1 else 0
        )
        for i in range(self.num_sensors):
            sensor_angle = self.angle - half_fov + i * angle_gap
            sensor_end_x = self.x + self.sensor_length * math.cos(
                math.radians(sensor_angle)
            )
            sensor_end_y = self.y - self.sensor_length * math.sin(
                math.radians(sensor_angle)
            )
            self.sensors.append((sensor_angle, (sensor_end_x, sensor_end_y)))

    def update_sensors(self):
        """Update sensor positions."""
        self.create_sensors()

    def get_target_angle(self):
        """
        Calculate the angle towards the current waypoint.

        Returns:
            float: The angle in degrees towards the target waypoint.
        """
        if self.current_waypoint_index is None:
            return self.angle  # If no waypoint selected, return current angle

        target_x, target_y = self.waypoints[self.current_waypoint_index]
        dx = target_x - self.x
        dy = self.y - target_y  # Inverted y-axis for Pygame
        angle = math.degrees(math.atan2(dy, dx))
        return angle % 360

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
            self.angle = target_angle
        elif angle_diff > 0:
            self.angle += CAR_ROTATION_SPEED
        else:
            self.angle -= CAR_ROTATION_SPEED

        self.angle %= 360
        self.update_sensors()

    def move_forward(self):
        """Move the car forward based on its speed and angle."""
        new_x = self.x + self.speed * math.cos(math.radians(self.angle))
        new_y = self.y - self.speed * math.sin(math.radians(self.angle))
        if not self.check_collision(new_x, new_y):
            self.x = new_x
            self.y = new_y
        else:
            logger.warning("Collision detected! Movement blocked.")

    def check_waypoint_reached(self):
        """Check if the current waypoint has been reached."""
        if self.current_waypoint_index is None:
            return False

        target_x, target_y = self.waypoints[self.current_waypoint_index]
        distance = math.hypot(target_x - self.x, target_y - self.y)
        if distance < self.threshold:
            logger.info(
                f"Reached waypoint {self.waypoint_names[self.current_waypoint_index]}: ({target_x}, {target_y})"
            )
            self.moving = False  # Stop moving after reaching the waypoint

            if not self.is_returning_to_start:
                # Only set awaiting_choice if not returning to start
                self.awaiting_choice = True
                self.state_reason = f"Reached {self.waypoint_names[self.current_waypoint_index]}"  # Update reason
            else:
                # If returning to start, reset the flag
                self.is_returning_to_start = False
                self.state_reason = "At Start Point"

            self.current_waypoint_index = None  # No active waypoint now
            return True
        return False

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
            self.x, self.y = original_x, original_y
        return collision

    def return_to_start(self):
        """Move the car back to the starting position using path following."""
        self.is_returning_to_start = True
        self.current_waypoint_index = 0  # Set waypoint index to start
        self.moving = True  # Start moving back to start
        self.state_reason = "Returning to start point"  # Update reason

    def update(self):
        """Update the car's state by rotating and moving towards the target."""
        if self.moving and not self.obstacle_detected:  # Only move if no obstacle
            target_angle = self.get_target_angle()
            self.rotate_towards_target(target_angle)
            self.move_forward()
            self.check_waypoint_reached()

    def draw_status(self, surface):
        """Draw the current status and reason for stop/move on the screen."""
        font = pygame.font.SysFont(None, 36)
        status = "MOVING" if self.moving else "STOPPED"
        status_text = font.render(f"Car Status: {status}", True, BLACK)
        reason_text = font.render(f"Reason: {self.state_reason}", True, BLACK)

        # Get width of the status text to dynamically place the reason text
        status_text_width = status_text.get_width()

        # Draw status and reason next to each other
        surface.blit(status_text, (10, 10))  # Status at top-left
        surface.blit(
            reason_text, (10 + status_text_width + 20, 10)
        )  # Reason beside the status with 20px gap

    def draw(self, surface):
        """Render the car, sensors, and waypoints on the screen."""
        rotated_image, rect = self.update_mask()
        surface.blit(rotated_image, rect.topleft)

        # Draw sensors
        for sensor_angle, (sensor_end_x, sensor_end_y) in self.sensors:
            pygame.draw.line(
                surface, RED, (self.x, self.y), (sensor_end_x, sensor_end_y), 2
            )
            pygame.draw.circle(surface, RED, (int(sensor_end_x), int(sensor_end_y)), 5)

        # Draw waypoints
        for idx, (wp_x, wp_y) in enumerate(self.waypoints):
            color = GREEN if idx == self.current_waypoint_index else BLUE
            pygame.draw.circle(surface, color, (int(wp_x), int(wp_y)), 8)
            font = pygame.font.SysFont(None, 24)
            img = font.render(self.waypoint_names[idx], True, BLACK)
            surface.blit(img, (wp_x + 10, wp_y - 10))


# -------------------- SerialReader Class ---------------------#


class SerialReader(threading.Thread):
    """Handles serial communication with the Arduino."""

    def __init__(self, serial_port, baud_rate, car):
        """
        Initialize the SerialReader thread.

        Args:
            serial_port (str): The serial port to connect to.
            baud_rate (int): The baud rate for communication.
            car (CarRobot): The car object to control based on serial data.
        """
        super().__init__()
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.running = True
        self.ser = None
        self.car = car  # Car instance to control
        self.state = "MOVING"  # Default state
        self.lock = threading.Lock()

    def run(self):
        """Run the thread to continuously read from the serial port."""
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
                                self.car.obstacle_detected = (
                                    True  # Set obstacle detected flag
                                )
                                self.car.moving = False  # Stop car movement
                                self.car.state_reason = "Obstacle detected"
                            elif state == "MOVING":
                                if self.state != "MOVING":
                                    logger.info("Arduino: MOVING")
                                self.state = "MOVING"
                                self.car.obstacle_detected = (
                                    False  # Clear obstacle detected flag
                                )
                                if self.car.current_waypoint_index is not None:
                                    self.car.moving = (
                                        True  # Resume movement if waypoint is set
                                    )
                                    self.car.state_reason = "Moving towards waypoint"
        except serial.SerialException as e:
            logger.error(f"Serial Exception: {e}")
            self.running = False

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
        pygame.display.set_caption("Autonomous Car Navigation")
        self.clock = pygame.time.Clock()
        self.walls = self.create_walls()
        self.define_waypoints()
        self.waypoint_names = ["Start", "M215", "M216", "M217"]
        self.car = CarRobot(
            self.waypoints[0][0],
            self.waypoints[0][1],
            self.waypoints,
            self.waypoint_names,
            self.walls,
        )

        # Initialize serial communication for obstacle detection
        self.serial_reader = SerialReader(SERIAL_PORT, BAUD_RATE, self.car)
        self.serial_reader.start()

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
            pygame.Rect(700, 50, 50, 500),  # Right wall
        ]
        return [Wall(rect) for rect in wall_rects]

    def define_waypoints(self):
        """
        Define the initial waypoints for the car to navigate.
        """
        self.waypoints = [
            (150, 150),  # Start
            (650, 150),  # M215
            (650, 450),  # M216
            (150, 450),  # M217
        ]

    def draw_walls(self):
        """Draw all walls on the screen."""
        for wall in self.walls:
            wall.draw(self.screen)

    def choose_waypoint(self, mouse_x, mouse_y):
        """Choose a waypoint based on mouse click and update the car's target."""
        closest_index = None
        min_distance = float("inf")

        for idx, (wp_x, wp_y) in enumerate(self.waypoints):
            distance = math.hypot(mouse_x - wp_x, mouse_y - wp_y)
            if (
                distance < min_distance and distance < 50
            ):  # Threshold for clicking accuracy
                closest_index = idx
                min_distance = distance

        if closest_index is not None:
            self.car.current_waypoint_index = closest_index
            self.car.moving = True
            self.car.state_reason = (
                f"Moving towards {self.car.waypoint_names[closest_index]}"
            )
            logger.info(
                f"Selected waypoint {self.car.waypoint_names[closest_index]}: ({self.waypoints[closest_index][0]}, {self.waypoints[closest_index][1]})"
            )

    def handle_choice(self):
        """Handle the choice between selecting another waypoint or returning to start."""
        font = pygame.font.SysFont(None, 36)

        # First, redraw the environment (2D map, car, etc.)
        self.screen.fill(WHITE)
        self.draw_walls()
        self.car.draw(self.screen)
        self.car.draw_status(self.screen)

        # Define padding and spacing
        padding = 20
        spacing = 20

        # Define maximum width for prompt text
        max_prompt_width = (
            WIDTH - (3 * padding) - (2 * 160) - (2 * spacing)
        )  # Total button widths and spacing
        prompt_text = "Are you done or go to another place?"

        # Wrap the prompt text into multiple lines if necessary
        prompt_lines = []
        current_line = ""
        for word in prompt_text.split():
            test_line = f"{current_line} {word}".strip()
            test_width, _ = font.size(test_line)
            if test_width <= max_prompt_width:
                current_line = test_line
            else:
                if current_line:
                    prompt_lines.append(current_line)
                current_line = word
        if current_line:
            prompt_lines.append(current_line)

        # Render each line of prompt text
        for idx, line in enumerate(prompt_lines):
            rendered_text = font.render(line, True, BLACK)
            text_rect = rendered_text.get_rect()
            text_rect.topleft = (
                padding,
                HEIGHT + padding + idx * (font.get_height() + 5),
            )
            self.screen.blit(rendered_text, text_rect)

        # Calculate the vertical position for buttons based on the number of prompt lines
        buttons_y = (
            HEIGHT + padding + len(prompt_lines) * (font.get_height() + 5) + spacing
        )

        # Define button dimensions
        button_width = 160
        button_height = 40

        # Define button positions horizontally with spacing
        go_button_x = padding
        done_button_x = go_button_x + button_width + spacing

        # Ensure buttons do not exceed window width
        if done_button_x + button_width + padding > WIDTH:
            # Align the "Done" button to the right with padding
            done_button_x = WIDTH - padding - button_width

        # Y position for buttons (centered vertically in button area)
        button_y = buttons_y

        # Define button rectangles with borders
        go_button_rect = pygame.Rect(go_button_x, button_y, button_width, button_height)
        done_button_rect = pygame.Rect(
            done_button_x, button_y, button_width, button_height
        )

        # Draw buttons (light gray for "Go Another" and dark gray for "Done") with borders
        pygame.draw.rect(self.screen, LIGHT_GRAY, go_button_rect)
        pygame.draw.rect(self.screen, DARK_GRAY, done_button_rect)
        pygame.draw.rect(
            self.screen, BLACK, go_button_rect, 2
        )  # Border for "Go Another"
        pygame.draw.rect(self.screen, BLACK, done_button_rect, 2)  # Border for "Done"

        # Render button text
        go_text = font.render("Go Another", True, BLACK)
        done_text = font.render("Done", True, BLACK)
        go_text_rect = go_text.get_rect(center=go_button_rect.center)
        done_text_rect = done_text.get_rect(center=done_button_rect.center)

        # Blit the text onto the buttons
        self.screen.blit(go_text, go_text_rect)
        self.screen.blit(done_text, done_text_rect)

        # Update the screen to display buttons and text
        pygame.display.flip()

        # Wait for user choice (either Go Another or Done)
        waiting_for_choice = True
        while waiting_for_choice:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()

                    # Check if "Go Another" button is clicked
                    if go_button_rect.collidepoint(mouse_x, mouse_y):
                        logger.info("Choosing another waypoint.")
                        waiting_for_choice = False
                        self.car.awaiting_choice = False

                    # Check if "Done" button is clicked
                    elif done_button_rect.collidepoint(mouse_x, mouse_y):
                        logger.info("Returning to Start.")
                        self.car.return_to_start()  # Start the car's journey to the start point
                        waiting_for_choice = False
                        self.car.awaiting_choice = False

    def run(self):
        """Main game loop."""
        running = True
        while running:
            self.screen.fill(WHITE)

            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    if (
                        mouse_y < HEIGHT
                    ):  # Only allow selecting waypoints within the map area
                        self.choose_waypoint(mouse_x, mouse_y)

            # Update car movement
            if not self.car.awaiting_choice:
                self.car.update()

            # If a waypoint was reached, show choice prompt
            if self.car.awaiting_choice:
                # Only show the choice prompt if the car hasn't returned to the start
                if not self.car.is_returning_to_start:
                    self.handle_choice()

            # Render environment and car
            self.draw_walls()
            self.car.draw(self.screen)
            self.car.draw_status(self.screen)  # Draw car status on the screen

            # Update the display and tick the clock
            pygame.display.flip()
            self.clock.tick(FPS)

        # Clean up and stop serial reader thread
        self.serial_reader.stop()
        self.serial_reader.join()

        pygame.quit()
        sys.exit()


# -------------------- Main Block ---------------------#


if __name__ == "__main__":
    try:
        game = Game()
        game.run()
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
