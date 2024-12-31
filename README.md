# Space Invaders with YOLO Pose Detection

This project is a modern take on the classic Space Invaders game, where the player's ship is controlled using YOLO pose detection via a webcam. The game leverages the YOLOv8 model for real-time pose estimation, with the player's nose position used to control the ship's movement.

## Features
- **Real-time pose detection**: Use your webcam to control the ship with your nose.
- **Dynamic gameplay**: Enemies spawn at random, and you auto-fire bullets to defeat them.
- **Explosions and effects**: Enemies explode with visual effects when hit.
- **Score tracking**: Your score is displayed in real-time.

## Requirements
- Python 3.8+
- A GPU is recommended for smoother YOLO detection.
- A webcam for pose detection.

### Python Dependencies
The required dependencies are listed in `requirements.txt`:
```txt
cv2
numpy
pygame
ultralytics
```
Install them using:
```bash
pip install -r requirements.txt
```

## Getting Started

### 1. Clone the Repository
```bash
git clone Core-Pose-Estimation-Collection.git
cd Core-Pose-Estimation-Collection
```

### 2. Setup the Environment
Ensure you have Python 3.8 or higher installed. Install the required dependencies as described above.

### 3. Download YOLO Model
This project uses the `yolo11n-pose.pt` model. It will be downloaded when program is run. After initial download the program will preload the model.

### 4. Run the Game
Run the script to start the game (on GPU):
```bash
python CpuSpaceInvaders.py
```

If you have setup cuda on your device, run the script to start the game (on GPU):
```bash
python Space_invaders.py
```


## How to Play
1. Ensure your webcam is functioning.
2. Use your nose to move the ship horizontally across the screen.
3. Enemies spawn at the top of the screen and move downward. Your ship auto-fires bullets to destroy them.
4. Avoid letting enemies reach the bottom of the screen.
5. Aim for the highest score possible!

## Project Structure
- `CpuSpaceInvaders` and `space_invaders.py`: Main game scripts.
- `requirements.txt`: List of dependencies.
- `yolo11n-pose.pt`: YOLO pose detection model file (not included; download separately).

## Controls
- **Movement**: Controlled via the position of your nose detected by YOLO.
- **Quit**: Press `Q` or close the game window.

## Known Issues
- If the webcam cannot be accessed, the program will raise a runtime error.
- Ensure the YOLO model file is correctly loaded to avoid detection errors.

## Future Enhancements
- Add more advanced enemy patterns and levels.
- Introduce power-ups and additional ship features.
- Optimize performance for slower systems.

## Acknowledgements
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the YOLOv8 model.
- [Pygame](https://www.pygame.org/) for game development tools.
Special thanks to the open-source community for providing the tools and resources that made this project possible.

