# ============================= #
#            IMPORTS             #
# ============================= #

import pygame
import math
import sys
import serial
import threading

# Initialize Pygame
pygame.init()

# ============================= #
#           CONSTANTS            #
# ============================= #

WIDTH, HEIGHT = 800, 600

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

CAR_IMAGE_PATH = "navigationTry/2d-super-car-top-view.png"
CAR_SIZE = (100, 50)
CAR_SPEED = 2
CAR_ROTATION_SPEED = 5

NUM_SENSORS = 3
SENSOR_LENGTH = 150
SENSOR_FOV = 30

WAYPOINT_THRESHOLD = 20

FPS = 60

SERIAL_PORT = "COM5"  # Replace with your Arduino's serial port
BAUD_RATE = 9600

# ============================= #
#            CLASSES             #
# ============================= #


class Wall:
    """Represents a wall in the game environment."""

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
    """Represents the autonomous car robot."""

    def __init__(self, x, y, waypoints, walls):
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = CAR_SPEED
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.direction = 1
        self.threshold = WAYPOINT_THRESHOLD
        self.walls = walls
        self.sensors = []
        self.num_sensors = NUM_SENSORS
        self.sensor_length = SENSOR_LENGTH
        self.sensor_fov = SENSOR_FOV
        self.load_image()
        self.create_sensors()

    def load_image(self):
        self.original_image = pygame.image.load(CAR_IMAGE_PATH)
        self.original_image = pygame.transform.scale(self.original_image, CAR_SIZE)

    def create_sensors(self):
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
        self.create_sensors()

    def get_target_angle(self):
        target_x, target_y = self.waypoints[self.current_waypoint_index]
        dx = target_x - self.x
        dy = self.y - target_y
        angle = math.degrees(math.atan2(dy, dx))
        return angle % 360

    def rotate_towards_target(self, target_angle):
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
        new_x = self.x + self.speed * math.cos(math.radians(self.angle))
        new_y = self.y - self.speed * math.sin(math.radians(self.angle))
        if not self.check_collision(new_x, new_y):
            self.x = new_x
            self.y = new_y
        else:
            print("Collision detected! Movement blocked.")

    def check_waypoint_reached(self):
        target_x, target_y = self.waypoints[self.current_waypoint_index]
        distance = math.hypot(target_x - self.x, target_y - self.y)
        if distance < self.threshold:
            print(
                f"Reached waypoint {self.current_waypoint_index + 1}: ({target_x}, {target_y})"
            )
            self.current_waypoint_index += self.direction

            if self.current_waypoint_index >= len(self.waypoints):
                self.current_waypoint_index = len(self.waypoints) - 2
                self.direction = -1
                print("Reversing direction to backward.")
            elif self.current_waypoint_index < 0:
                self.current_waypoint_index = 1
                self.direction = 1
                print("Reversing direction to forward.")

    def update_mask(self):
        rotated_image = pygame.transform.rotate(self.original_image, self.angle)
        self.rotated_rect = rotated_image.get_rect(center=(self.x, self.y))
        self.car_mask = pygame.mask.from_surface(rotated_image)
        return rotated_image, self.rotated_rect

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
                print(f"Collision with wall at position: {wall.rect}")
                collision = True
                break

        if collision:
            self.x, self.y = original_x, original_y
        return collision

    def update(self):
        target_angle = self.get_target_angle()
        self.rotate_towards_target(target_angle)
        self.move_forward()
        self.check_waypoint_reached()

    def draw_state(self, surface, state):
        font = pygame.font.SysFont(None, 36)
        state_text = font.render(f"State: {state}", True, BLACK)
        surface.blit(state_text, (10, 10))

    def draw(self, surface, state):
        rotated_image, rect = self.update_mask()
        surface.blit(rotated_image, rect.topleft)

        for sensor_angle, (sensor_end_x, sensor_end_y) in self.sensors:
            pygame.draw.line(
                surface, RED, (self.x, self.y), (sensor_end_x, sensor_end_y), 2
            )
            pygame.draw.circle(surface, RED, (int(sensor_end_x), int(sensor_end_y)), 5)

        for idx, (wp_x, wp_y) in enumerate(self.waypoints):
            color = GREEN if idx == self.current_waypoint_index else BLUE
            pygame.draw.circle(surface, color, (int(wp_x), int(wp_y)), 8)
            font = pygame.font.SysFont(None, 24)
            img = font.render(str(idx + 1), True, BLACK)
            surface.blit(img, (wp_x + 10, wp_y - 10))

        self.draw_state(surface, state)


class SerialReader(threading.Thread):
    """Thread to read serial data from Arduino."""

    def __init__(self, serial_port, baud_rate):
        super().__init__()
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.running = True
        self.ser = None
        self.state = "MOVING"
        self.lock = threading.Lock()

    def run(self):
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
        with self.lock:
            return self.state

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()


class Game:
    """Represents the main game."""

    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Autonomous Car Navigation")
        self.clock = pygame.time.Clock()
        self.walls = self.create_walls()
        self.waypoints = self.define_waypoints()
        self.car = CarRobot(
            self.waypoints[0][0], self.waypoints[0][1], self.waypoints, self.walls
        )

        self.serial_reader = SerialReader(SERIAL_PORT, BAUD_RATE)
        self.serial_reader.start()

        self.is_moving = True

    def create_walls(self):
        wall_rects = [
            pygame.Rect(50, 50, 700, 50),
            pygame.Rect(50, 500, 700, 50),
            pygame.Rect(50, 50, 50, 500),
            pygame.Rect(700, 50, 50, 500),
        ]
        return [Wall(rect) for rect in wall_rects]

    def define_waypoints(self):
        return [
            (150, 150),
            (650, 150),
            (650, 450),
            (150, 450),
        ]

    def draw_walls(self):
        for wall in self.walls:
            wall.draw(self.screen)

    def run(self):
        running = True
        while running:
            self.screen.fill(WHITE)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    self.waypoints.append((mouse_x, mouse_y))
                    self.car.waypoints = self.waypoints
                    print(f"New waypoint added: ({mouse_x}, {mouse_y})")

            self.is_moving = self.serial_reader.get_state() == "MOVING"
            current_state = "MOVING" if self.is_moving else "STOPPED"
            print(f"Car movement state: {current_state}")

            if self.is_moving:
                self.car.update()
            else:
                print("CarRobot: Movement paused due to obstacle.")

            self.draw_walls()
            self.car.draw(self.screen, current_state)

            pygame.display.flip()
            self.clock.tick(FPS)

        self.serial_reader.stop()
        self.serial_reader.join()
        pygame.quit()
        sys.exit()


# ============================= #
#             MAIN               #
# ============================= #

if __name__ == "__main__":
    game = Game()
    game.run()
