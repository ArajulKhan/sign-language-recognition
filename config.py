"""Central configuration. Edit ACTIONS to define as your wish sign vocabulary."""

from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "sign_model.keras"
LABELS_PATH = MODEL_DIR / "labels.json"

# ---- Vocabulary -----------------------------------------------------------
# "idle" = no sign / hands resting / random movement. It is REQUIRED: it teaches
# the model when nobody is signing, which prevents false words.
# "clear" = a sign you choose that erases the sentence.
ACTIONS = [
    "idle",
    "hello", "my", "name is", "you", "how", "i love you",
    "thanks", "please", "help", "yes", "no",
    "drink", "eat", "clear",
]

# ---- Data collection ------------------------------------------------------
SEQ_LEN = 30          # frames per sample (~1 second at 30 FPS)
NUM_SEQUENCES = 40    # samples per sign (use 60+ for better results)
START_DELAY_MS = 1000 # pause before each recording
CAMERA_INDEX = 0

# ---- Features -------------------------------------------------------------
# per hand: 21 landmarks * 3 (normalized) + 2 (wrist position) = 65
FEATURES_PER_HAND = 65
NUM_FEATURES = FEATURES_PER_HAND * 2

# ---- Training -------------------------------------------------------------
EPOCHS = 80
BATCH_SIZE = 32
TEST_SIZE = 0.2
AUGMENT_COPIES = 3

# ---- Real-time inference --------------------------------------------------
CONF_THRESHOLD = 0.85   # minimum probability to accept a prediction
STABLE_N = 6            # consecutive identical predictions needed to commit a word
MAX_NO_HAND = 10        # frames without hands before treating state as idle
PREDICT_EVERY = 2       # run the model every N frames
MAX_WORDS_SHOWN = 8
