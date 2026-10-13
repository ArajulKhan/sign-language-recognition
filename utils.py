"""MediaPipe hand detection (Tasks API / HandLandmarker) + feature extraction.
 
Works with newer MediaPipe versions where `mp.solutions` no longer exists.
Shared by collect_data.py, train.py and realtime.py. The public function names
are the same as the old version, so the other scripts need no changes.
"""
import time
import urllib.request
 
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
 
import config
 
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_MODEL_PATH = config.MODEL_DIR / "hand_landmarker.task"
 
# Standard 21-point hand skeleton (drawn manually, since drawing_utils is gone)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm edge
]
 
 
def ensure_hand_model():
    """Download the hand landmark model once (about 7 MB) if it's missing."""
    if HAND_MODEL_PATH.exists() and HAND_MODEL_PATH.stat().st_size > 0:
        return HAND_MODEL_PATH
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading hand_landmarker.task ...")
    try:
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    except Exception as e:
        HAND_MODEL_PATH.unlink(missing_ok=True)
        raise SystemExit(
            f"Could not download the hand model ({e}).\n"
            f"Download it manually from:\n  {HAND_MODEL_URL}\n"
            f"and save it as:\n  {HAND_MODEL_PATH}"
        )
    return HAND_MODEL_PATH
 
 
class HandTracker:
    """Thin wrapper so it can be used as: with create_hands() as hands: ..."""
 
    def __init__(self):
        model_bytes = ensure_hand_model().read_bytes()
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = 0
 
    def detect(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        # timestamps must strictly increase in VIDEO mode
        ts = max(self._last_ts + 1, int(time.monotonic() * 1000))
        self._last_ts = ts
        return self._landmarker.detect_for_video(mp_image, ts)
 
    def close(self):
        self._landmarker.close()
 
    def __enter__(self):
        return self
 
    def __exit__(self, *exc):
        self.close()
 
 
def create_hands():
    return HandTracker()
 
 
def detect(frame_bgr, hands):
    return hands.detect(frame_bgr)
 
 
def hands_present(results) -> bool:
    return len(results.hand_landmarks) > 0
 
 
def _normalize_hand(landmarks) -> np.ndarray:
    """Wrist-relative, scale-invariant landmarks + absolute wrist (x, y).
 
    Relative coordinates make the shape independent of position/distance;
    the absolute wrist keeps *where* the sign happens (important in sign language).
    """
    pts = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    wrist = pts[0].copy()
    rel = pts - wrist
    scale = np.linalg.norm(pts[9] - pts[0]) + 1e-6  # wrist -> middle finger base
    rel /= scale
    return np.concatenate([rel.flatten(), wrist[:2]])
 
 
def extract_features(results) -> np.ndarray:
    """Return a fixed-length vector (2 hands x 65). Missing hand = zeros."""
    slots = [np.zeros(config.FEATURES_PER_HAND, np.float32) for _ in range(2)]
    if results.hand_landmarks:
        hands = [_normalize_hand(h) for h in results.hand_landmarks]
        if len(hands) == 1:
            label = results.handedness[0][0].category_name
            slots[0 if label == "Left" else 1] = hands[0]
        else:
            # sort by horizontal wrist position -> stable left/right slot assignment
            hands.sort(key=lambda f: f[-2])
            slots[0], slots[1] = hands[0], hands[1]
    return np.concatenate(slots)
 
 
def draw_hands(frame, results):
    h, w = frame.shape[:2]
    for landmarks in results.hand_landmarks:
        pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 4, (0, 0, 255), -1, cv2.LINE_AA)
    return frame
 
 
def put_text_box(frame, text, org, scale=0.9, color=(255, 255, 255), bg=(0, 0, 0), thickness=2):
    """Draw text with a filled background so it stays readable."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 6, y - h - 6), (x + w + 6, y + base + 4), bg, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
 
