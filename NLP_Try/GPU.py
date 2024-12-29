import pygame
import sys
import json

# Initialize Pygame
pygame.init()

# Real-world dimensions (meters)
inner_points_real = [
    (0, 0),  # Start point
    (0, 20),  # First corner
    (39.3, 20),  # Second corner
    (39.3, 0),  # Last point
]

outer_points_real = [
    (-2.7, 0),  # Start point
    (-2.7, 23.5),  # First corner
    (42.8, 23.5),  # Second corner
    (42.8, 0),  # Last point
]

# Maximum window dimensions
MAX_SCREEN_WIDTH = 800
MAX_SCREEN_HEIGHT = 600

# Real-world dimensions
real_width = 42.8 + 2.7  # Outer width in meters
real_height = 23.5  # Outer height in meters

# Normalize points to start at (0, 0)
normalized_inner_points = [(x + 2.7, y) for x, y in inner_points_real]
normalized_outer_points = [(x + 2.7, y) for x, y in outer_points_real]

# Calculate scaling factor to fit within the window
SCALE_WIDTH = MAX_SCREEN_WIDTH / real_width
SCALE_HEIGHT = MAX_SCREEN_HEIGHT / real_height
SCALE = min(SCALE_WIDTH, SCALE_HEIGHT) * 0.85  # Add padding (85% of available space)

# Screen dimensions after scaling
corridor_width = int(real_width * SCALE)
corridor_height = int(real_height * SCALE)

# Add padding around the corridor
FRAME_PADDING = 20  # Space between corridor and frame
frame_width = corridor_width + 2 * FRAME_PADDING
frame_height = corridor_height + 2 * FRAME_PADDING

# Calculate offsets to perfectly center the frame
offset_x = (MAX_SCREEN_WIDTH - frame_width) // 2
offset_y = (MAX_SCREEN_HEIGHT - frame_height) // 2

# Scale the normalized points for visualization
inner_points_scaled = [
    (x * SCALE, corridor_height - y * SCALE) for x, y in normalized_inner_points
]
outer_points_scaled = [
    (x * SCALE, corridor_height - y * SCALE) for x, y in normalized_outer_points
]

# Apply offsets to center the corridor and padding
inner_points_scaled = [
    (x + offset_x + FRAME_PADDING, y + offset_y + FRAME_PADDING)
    for x, y in inner_points_scaled
]
outer_points_scaled = [
    (x + offset_x + FRAME_PADDING, y + offset_y + FRAME_PADDING)
    for x, y in outer_points_scaled
]

# Save the real-world map as JSON for robot use
map_data = {"inner_points": inner_points_real, "outer_points": outer_points_real}

with open("robot_map.json", "w") as f:
    json.dump(map_data, f, indent=4)

print("Real-world map saved to robot_map.json")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)  # Boundary lines
BLUE = (0, 0, 255)  # Frame for padding

# Main loop for Pygame visualization
screen = pygame.display.set_mode((MAX_SCREEN_WIDTH, MAX_SCREEN_HEIGHT))
pygame.display.set_caption("Scaled Corridor Map")
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill screen with white
    screen.fill(WHITE)

    # Draw the frame for padding
    pygame.draw.rect(
        screen,
        BLUE,
        pygame.Rect(
            offset_x,
            offset_y,
            frame_width,
            frame_height,
        ),
        2,  # Line thickness
    )

    # Draw the outer boundary (same color as inner boundary)
    pygame.draw.polygon(screen, BLACK, outer_points_scaled, 2)

    # Draw the inner boundary (same color as outer boundary)
    pygame.draw.polygon(screen, BLACK, inner_points_scaled, 2)

    # Connect corresponding points of inner and outer boundaries (black lines)
    for i in range(len(inner_points_real)):
        pygame.draw.line(
            screen, BLACK, inner_points_scaled[i], outer_points_scaled[i], 1
        )

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
