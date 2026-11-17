from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
INPUT_VIDEO = ROOT_DIR / "gameplay.mp4"
OUTPUT_VIDEO = ROOT_DIR / "output_preview.mp4"

MODEL_WEIGHTS = ROOT_DIR / "yolo11n.pt"
# Allow weaker detections to support existing tracks. New tracks have a higher threshold.
DETECTION_CONFIDENCE = 0.10
INFERENCE_SIZE = 640
CUDA_DEVICE = 0
TRACKER_CONFIG = str(ROOT_DIR / "tracker_config.yaml")
EMA_ALPHA = 0.55
EMA_MAX_GAP_FRAMES = 8

DONOR_LABELS = (
    "Alex_UA • $50",
    "ViceKing • $100",
    "NeonRider • $25",
    "LuciaFan • $10",
    "Donator_X • $200",
    "OceanDrive • $35",
    "PalmCity • $15",
    "MidnightFM • $75",
)

# OpenCV uses BGR color order.
BADGE_COLORS = (
    (255, 219, 92),
    (193, 143, 255),
    (132, 239, 118),
    (98, 203, 255),
    (220, 148, 255),
    (222, 236, 116),
    (115, 172, 255),
    (241, 173, 161),
)

BADGE_HEAD_GAP = 12
WINDOW_NAME = "NPC Overlay Preview"
OUTPUT_FOURCC = "mp4v"
