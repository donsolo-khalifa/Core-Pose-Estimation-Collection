import cv2
from ultralytics import YOLO
import pygame
import numpy as np
import random
import math
from threading import Thread
from queue import Queue

# Initialize YOLO model
model = YOLO("../yolo11n-pose.pt")

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

# Game objects
SHIP_WIDTH = int(cam_width * 0.05)
SHIP_HEIGHT = int(cam_width * 0.05)
ship_x = cam_width // 2
bullets = []
BULLET_SPEED = cam_height * 0.02
BULLET_RADIUS = int(cam_width * 0.005)
enemies = []
ENEMY_WIDTH = int(cam_width * 0.05)
ENEMY_HEIGHT = int(cam_width * 0.05)
ENEMY_SPEED = cam_height * 0.005
explosions = []

# Shared variables and queues
frame_queue = Queue(maxsize=1)
results_queue = Queue(maxsize=1)
running = True


def process_frames():
    """Thread for processing frames with YOLO."""
    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip frame horizontally
        frame = cv2.flip(frame, 1)

        # Run YOLO detection
        results = model.predict(frame, verbose=False)
        if not results_queue.full():
            results_queue.put((frame, results))


def get_keypoint_position(results):
    """Extract nose position from YOLO results."""
    try:
        keypoints = results[0].keypoints.xyn[0]
        nose_x = keypoints[0][0].item()
        return nose_x
    except (IndexError, AttributeError):
        return None


def draw_enemy(x, y, surface):
    """Draw an enemy ship."""
    points = [
        (x + ENEMY_WIDTH // 2, y),
        (x + ENEMY_WIDTH, y + ENEMY_HEIGHT // 2),
        (x + ENEMY_WIDTH // 2, y + ENEMY_HEIGHT),
        (x, y + ENEMY_HEIGHT // 2)
    ]
    pygame.draw.polygon(surface, RED, points)

def draw_explosion(x, y, surface):
    """Draw explosion effect."""
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





# Start processing thread
Thread(target=process_frames, daemon=True).start()

# Game loop
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

        # Explosion handling
        for explosion in explosions[:]:
            if explosion['timer'] > 0:
                draw_explosion(explosion['x'], explosion['y'], screen)
                explosion['timer'] -= 1
            else:
                explosions.remove(explosion)
        # Check for processed frames
        if not results_queue.empty():
            frame, results = results_queue.get()
            nose_x = get_keypoint_position(results)

            if nose_x is not None:
                ship_x = int(np.interp(nose_x, [0, 1], [0, cam_width - SHIP_WIDTH]))

            # Convert frame to Pygame surface
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            frame = pygame.surfarray.make_surface(frame)
            frame = pygame.transform.flip(frame, True, False)
            screen.blit(frame, (0, 0))

        # Game logic
        if random.random() < 0.02:
            enemies.append({'x': random.randint(0, cam_width - ENEMY_WIDTH), 'y': 0})

        shoot_cooldown -= 1
        if shoot_cooldown <= 0:
            bullets.append({'x': ship_x + SHIP_WIDTH // 2, 'y': cam_height - SHIP_HEIGHT - 10})
            shoot_cooldown = 12

        for bullet in bullets[:]:
            bullet['y'] -= BULLET_SPEED
            if bullet['y'] < 0:
                bullets.remove(bullet)
            else:
                pygame.draw.circle(screen, BLUE, (int(bullet['x']), int(bullet['y'])), BULLET_RADIUS + 2)
                pygame.draw.circle(screen, WHITE, (int(bullet['x']), int(bullet['y'])), BULLET_RADIUS)

        for enemy in enemies[:]:
            enemy['y'] += ENEMY_SPEED
            if enemy['y'] > cam_height:
                enemies.remove(enemy)
                continue

            # Draw enemy
            draw_enemy(enemy['x'], enemy['y'], screen)

            # Collision detection
            enemy_center = (enemy['x'] + ENEMY_WIDTH // 2, enemy['y'] + ENEMY_HEIGHT // 2)
            for bullet in bullets[:]:
                distance = math.hypot(bullet['x'] - enemy_center[0], bullet['y'] - enemy_center[1])
                if distance < ENEMY_WIDTH // 2:
                    bullets.remove(bullet)
                    enemies.remove(enemy)
                    explosions.append({'x': enemy_center[0], 'y': enemy_center[1], 'timer': 10})
                    score += 10
                    break

        for explosion in explosions[:]:
            if explosion['timer'] > 0:
                explosion['timer'] -= 1
            else:
                explosions.remove(explosion)

        # Draw the player's ship
        pygame.draw.polygon(
            screen,
            GREEN,
            [
                (ship_x + SHIP_WIDTH // 2, cam_height - SHIP_HEIGHT - 10),
                (ship_x, cam_height - 10),
                (ship_x + SHIP_WIDTH, cam_height - 10)
            ]
        )

        # Draw the score
        font = pygame.font.Font(None, 48)
        score_text = font.render(f'SCORE: {score}', True, GREEN)
        screen.blit(score_text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

finally:
    cap.release()
    pygame.quit()
