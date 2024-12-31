import cv2
from ultralytics import YOLO
import pygame
import numpy as np
import random
import math

# Initialize YOLO model
model = YOLO("yolo11n-pose.pt")
model.to('cuda')  # Move model to GPU if available

# Initialize webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Could not access webcam")

# Get webcam dimensions
cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize Pygame to match webcam dimensions
pygame.init()
screen = pygame.display.set_mode((cam_width, cam_height))
pygame.display.set_caption("Webcam Space Invaders")

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (179, 0, 255)
BLUE = (164, 224, 0)
YELLOW = (120, 163, 0)

# Game objects scaled to webcam dimensions
SHIP_WIDTH = int(cam_width * 0.05)  # 5% of screen width
SHIP_HEIGHT = int(cam_width * 0.05)
ship_x = cam_width // 2

# Bullet properties
bullets = []
BULLET_SPEED = cam_height * 0.02  # Scale speed to screen height
BULLET_RADIUS = int(cam_width * 0.005)  # Scale radius to screen width

# Enemy properties
enemies = []
ENEMY_WIDTH = int(cam_width * 0.05)
ENEMY_HEIGHT = int(cam_width * 0.05)
ENEMY_SPEED = cam_height * 0.005  # Scale speed to screen height

# Explosions list
explosions = []


def get_keypoint_position(results):
    """Get nose position from YOLO results"""
    try:
        keypoints = results[0].keypoints.xyn[0]
        nose_x = keypoints[0][0].item()  # Nose X coordinate
        return nose_x
    except (IndexError, AttributeError):
        return None


def draw_ship(x, y, surface):
    """Draw the player's ship"""
    points = [
        (x + SHIP_WIDTH // 2, y),
        (x, y + SHIP_HEIGHT),
        (x + SHIP_WIDTH, y + SHIP_HEIGHT)
    ]
    pygame.draw.polygon(surface, GREEN, points)

    # Engine flames
    flame_points = [
        (x + SHIP_WIDTH // 2, y + SHIP_HEIGHT),
        (x + SHIP_WIDTH // 2 - 10, y + SHIP_HEIGHT + 10),
        (x + SHIP_WIDTH // 2 + 10, y + SHIP_HEIGHT + 10)
    ]
    flame_color = (random.randint(200, 255), random.randint(100, 150), 0)
    pygame.draw.polygon(surface, flame_color, flame_points)


def draw_enemy(x, y, surface):
    """Draw an enemy ship"""
    points = [
        (x + ENEMY_WIDTH // 2, y),
        (x + ENEMY_WIDTH, y + ENEMY_HEIGHT // 2),
        (x + ENEMY_WIDTH // 2, y + ENEMY_HEIGHT),
        (x, y + ENEMY_HEIGHT // 2)
    ]
    pygame.draw.polygon(surface, RED, points)

    # Wings
    wing_points_left = [
        (x, y + ENEMY_HEIGHT // 2),
        (x - ENEMY_WIDTH // 4, y + ENEMY_HEIGHT // 2),
        (x, y + ENEMY_HEIGHT // 3)
    ]
    wing_points_right = [
        (x + ENEMY_WIDTH, y + ENEMY_HEIGHT // 2),
        (x + ENEMY_WIDTH + ENEMY_WIDTH // 4, y + ENEMY_HEIGHT // 2),
        (x + ENEMY_WIDTH, y + ENEMY_HEIGHT // 3)
    ]
    pygame.draw.polygon(surface, BLUE, wing_points_left)
    pygame.draw.polygon(surface, BLUE, wing_points_right)


def draw_explosion(x, y, surface):
    """Draw explosion effect"""
    max_radius = ENEMY_WIDTH
    num_shapes = 8

    for i in range(num_shapes):
        angle = (2 * math.pi * i) / num_shapes
        points = []
        for j in range(3):
            point_angle = angle + (2 * math.pi * j) / 3
            radius = max_radius * (1 - random.random() * 0.3)
            point_x = x + math.cos(point_angle) * radius
            point_y = y + math.sin(point_angle) * radius
            points.append((point_x, point_y))
        color = (255, max(100, 255 - i * 20), 0)
        pygame.draw.polygon(surface, color, points)


def create_enemy():
    """Create a new enemy at the top of the screen"""
    return {
        'x': random.randint(0, cam_width - ENEMY_WIDTH),
        'y': 0
    }


def fire_bullet(x, y):
    """Create a new bullet"""
    bullets.append({
        'x': x + SHIP_WIDTH // 2,
        'y': y
    })


# Game loop
running = True
clock = pygame.time.Clock()
score = 0
shoot_cooldown = 0

try:
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        # Capture and process webcam frame
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Run YOLO detection
        results = model.predict(frame, verbose=False)
        nose_x = get_keypoint_position(results)

        if nose_x is not None:
            ship_x = int(np.interp(nose_x, [0, 1], [0, cam_width - SHIP_WIDTH]))

        # Convert frame to Pygame surface
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        frame = pygame.transform.flip(frame, True, False)

        # Draw frame as background
        screen.blit(frame, (0, 0))

        # Game logic
        if random.random() < 0.02:
            enemies.append(create_enemy())

        # Auto-fire bullets
        shoot_cooldown -= 1
        if shoot_cooldown <= 0:
            fire_bullet(ship_x, cam_height - SHIP_HEIGHT - 10)
            shoot_cooldown = 12

        # Update and draw bullets
        for bullet in bullets[:]:
            bullet['y'] -= BULLET_SPEED
            if bullet['y'] < 0:
                bullets.remove(bullet)
            else:
                pygame.draw.circle(screen, BLUE, (int(bullet['x']), int(bullet['y'])), BULLET_RADIUS + 2)
                pygame.draw.circle(screen, WHITE, (int(bullet['x']), int(bullet['y'])), BULLET_RADIUS)

        # Update and draw enemies
        for enemy in enemies[:]:
            enemy['y'] += ENEMY_SPEED
            if enemy['y'] > cam_height:
                enemies.remove(enemy)
                continue

            # Collision detection
            enemy_center = (enemy['x'] + ENEMY_WIDTH // 2, enemy['y'] + ENEMY_HEIGHT // 2)
            for bullet in bullets[:]:
                distance = math.hypot(bullet['x'] - enemy_center[0], bullet['y'] - enemy_center[1])
                if distance < ENEMY_WIDTH // 2:
                    if bullet in bullets:
                        bullets.remove(bullet)
                    if enemy in enemies:
                        enemies.remove(enemy)
                        explosions.append({'x': enemy_center[0], 'y': enemy_center[1], 'timer': 10})
                    score += 10
                    break

            draw_enemy(enemy['x'], enemy['y'], screen)

        # Draw explosions
        for explosion in explosions[:]:
            if explosion['timer'] > 0:
                draw_explosion(explosion['x'], explosion['y'], screen)
                explosion['timer'] -= 1
            else:
                explosions.remove(explosion)

        # Draw ship
        draw_ship(ship_x, cam_height - SHIP_HEIGHT - 10, screen)

        # Draw score with background for visibility
        font = pygame.font.Font(None, 48)
        score_text = font.render(f'SCORE: {score}', True, GREEN)
        score_bg = pygame.Surface((score_text.get_width() + 20, score_text.get_height() + 10))
        score_bg.fill((0, 0, 0))
        score_bg.set_alpha(128)
        screen.blit(score_bg, (10, 10))
        screen.blit(score_text, (20, 15))

        pygame.display.flip()
        clock.tick(60)

finally:
    # Cleanup
    cap.release()
    pygame.quit()