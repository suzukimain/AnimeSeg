# RGB color definitions for each segmentation class
COLORS = {
    'background': (0, 0, 0),
    'hair_main': (255, 0, 0),        # Main hair - bright red
    'hair_thin': (128, 0, 0),        # Thin hair - dark red
    'skin': (255, 220, 180),
    'face': (100, 150, 255),
    'clothes': (180, 0, 255),
    'right_eyebrow': (0, 255, 100),
    'left_eyebrow': (150, 255, 0),
    'nose': (255, 140, 0),
    'mouth': (255, 0, 150),
    'right_eye': (255, 255, 0),
    'left_eye': (0, 255, 255),
    'unknown': (64, 64, 64),         # Unknown/ignore - dark gray
}

# Explicit class ID mapping
CLASS_TO_ID = {
    'background': 0,
    'skin': 1,
    'face': 2,
    'hair_main': 3,       # Primary hair (thick)
    'left_eye': 4,
    'right_eye': 5,
    'left_eyebrow': 6,
    'right_eyebrow': 7,
    'nose': 8,
    'mouth': 9,
    'clothes': 10,
    'hair_thin': 11,      # Secondary hair (thin lines, ahoges)
    'unknown': 12,        # Background alternative (for clothes uncertainty)
}

ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
NUM_CLASSES = len(CLASS_TO_ID)
ID_TO_COLOR = {cls_id: COLORS[cls_name] for cls_name, cls_id in CLASS_TO_ID.items()}
