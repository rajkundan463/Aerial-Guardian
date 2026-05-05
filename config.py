# Model
MODEL_PATH = "yolov8n.pt"

#Detection tuning for drone
CONF_THRESHOLD = 0.2      # detect small people
IMG_SIZE = 1280           # high resolution for small objects

# Dataset path (CHANGE THIS TO YOUR PATH)
SEQUENCE_PATH = "D:/VisDrone2019-MOT-val/VisDrone2019-MOT-val/sequences/uav0000086_00000_v"

# Output
VIDEO_OUTPUT = "output/output_visdrone.mp4"

# Resize (important for FPS vs accuracy)
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720