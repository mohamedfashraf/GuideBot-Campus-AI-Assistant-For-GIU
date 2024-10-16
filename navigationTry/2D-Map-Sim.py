import pygame
import math
import sys
import serial
import threading

# -------------------- Constants ---------------------#

# Pygame Initialization
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600

# Color definitions
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

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

    def __init__(self, x, y, waypoints, walls):
        """
        Initialize the CarRobot.

        Args:
            x (float): Initial x-coordinate.
            y (float): Initial y-coordinate.
            waypoints (list): List of waypoint tuples.
            walls (list): List of Wall objects for collision detection.
        """
        self.x = x
        self.y = y
        self.angle = 0  # Facing right initially
        self.speed = CAR_SPEED
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.direction = 1  # 1 for forward, -1 for backward
        self.threshold = WAYPOINT_THRESHOLD
        self.walls = walls
        self.sensors = []
        self.num_sensors = NUM_SENSORS
        self.sensor_length = SENSOR_LENGTH
        self.sensor_fov = SENSOR_FOV
        self.load_image()
        self.create_sensors()

    def load_image(self):
        """Load and scale the car image."""
        self.original_image = pygame.image.load(CAR_IMAGE_PATH)
        self.original_image = pygame.transform.scale(self.original_image, CAR_SIZE)

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
            print("Collision detected! Movement blocked.")

    def check_waypoint_reached(self):
        """Check if the current waypoint has been reached."""
        target_x, target_y = self.waypoints[self.current_waypoint_index]
        distance = math.hypot(target_x - self.x, target_y - self.y)
        if distance < self.threshold:
            print(
                f"Reached waypoint {self.current_waypoint_index + 1}: ({target_x}, {target_y})"
            )
            self.current_waypoint_index += self.direction

            # Reverse direction if end of waypoints is reached
            if self.current_waypoint_index >= len(self.waypoints):
                self.current_waypoint_index = len(self.waypoints) - 2
                self.direction = -1
                print("Reversing direction to backward.")
            elif self.current_waypoint_index < 0:
                self.current_waypoint_index = 1
                self.direction = 1
                print("Reversing direction to forward.")

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
                print(f"Collision with wall at position: {wall.rect}")
                collision = True
                break

        if collision:
            self.x, self.y = original_x, original_y
        return collision

    def update(self):
        """Update the car's state by rotating and moving towards the target."""
        target_angle = self.get_target_angle()
        self.rotate_towards_target(target_angle)
        self.move_forward()
        self.check_waypoint_reached()

    def draw_state(self, surface, state):
        """
        Draw the current state of the car on the screen.

        Args:
            surface (pygame.Surface): The surface to draw on.
            state (str): The current state (e.g., "MOVING", "STOPPED").
        """
        font = pygame.font.SysFont(None, 36)
        state_text = font.render(f"State: {state}", True, BLACK)
        surface.blit(state_text, (10, 10))

    def draw(self, surface, state):
        """
        Render the car, sensors, and waypoints on the screen.

        Args:
            surface (pygame.Surface): The surface to draw on.
            state (str): The current state of the car.
        """
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
            img = font.render(str(idx + 1), True, BLACK)
            surface.blit(img, (wp_x + 10, wp_y - 10))

        # Draw current state
        self.draw_state(surface, state)


# -------------------- SerialReader Class ---------------------#


class SerialReader(threading.Thread):
    """Handles serial communication with the Arduino."""

    def __init__(self, serial_port, baud_rate):
        """
        Initialize the SerialReader thread.

        Args:
            serial_port (str): The serial port to connect to.
            baud_rate (int): The baud rate for communication.
        """
        super().__init__()
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.running = True
        self.ser = None
        self.state = "MOVING"  # Default state
        self.lock = threading.Lock()

    def run(self):
        """Run the thread to continuously read from the serial port."""
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            print(
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
                                self.state = "STOPPED"
                                print("Arduino: STOPPED")
                            elif state == "MOVING":
                                self.state = "MOVING"
                                print("Arduino: MOVING")
        except serial.SerialException as e:
            print(f"Serial Exception: {e}")
            self.running = False

    def get_state(self):
        """
        Get the current state received from Arduino.

        Returns:
            str: The current state ("MOVING" or "STOPPED").
        """
        with self.lock:
            return self.state

    def stop(self):
        """Stop the serial reader thread and close the serial connection."""
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()


# -------------------- Game Class ---------------------#


class Game:
    """Main game class handling the game loop and rendering."""

    def __init__(self):
        """Initialize the game, including screen, clock, walls, waypoints, and car."""
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Autonomous Car Navigation")
        self.clock = pygame.time.Clock()
        self.walls = self.create_walls()
        self.define_waypoints()
        self.car = CarRobot(
            self.waypoints[0][0], self.waypoints[0][1], self.waypoints, self.walls
        )

        # Initialize serial communication
        self.serial_reader = SerialReader(SERIAL_PORT, BAUD_RATE)
        self.serial_reader.start()

        self.is_moving = True  # Initial movement state

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
            (150, 150),
            (650, 150),
            (650, 450),
            (150, 450),
        ]

    def draw_walls(self):
        """Draw all walls on the screen."""
        for wall in self.walls:
            wall.draw(self.screen)

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
                    self.waypoints.append((mouse_x, mouse_y))
                    self.car.waypoints = self.waypoints
                    print(f"New waypoint added: ({mouse_x}, {mouse_y})")

            # Update movement state based on serial input
            self.is_moving = self.serial_reader.get_state() == "MOVING"
            current_state = "MOVING" if self.is_moving else "STOPPED"
            print(f"Car movement state: {current_state}")

            # Update car movement if allowed
            if self.is_moving:
                self.car.update()
            else:
                print("CarRobot: Movement paused due to obstacle.")

            # Render environment and car
            self.draw_walls()
            self.car.draw(self.screen, current_state)

            # Update the display and tick the clock
            pygame.display.flip()
            self.clock.tick(FPS)

        # Clean up on exit
        self.serial_reader.stop()
        self.serial_reader.join()
        pygame.quit()
        sys.exit()


# -------------------- Main Block ---------------------#

if __name__ == "__main__":
    game = Game()
    game.run()
