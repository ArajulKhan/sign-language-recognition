# AGENTS.md

Guidance for AI coding agents (and new contributors) working on this repository.

## Project summary

Real-time sign language recognition. A webcam feed goes through MediaPipe
HandLandmarker, is turned into a 130-number feature vector per frame, and a
30-frame window is classified by a Conv1D + BiLSTM model. Stable predictions are
assembled into a sentence drawn on the video frame.

Scope: a fixed vocabulary of signs (isolated-sign classification + sentence
assembly). It is NOT continuous sign recognition.

## Commands

```bash
pip install -e ".[dev]"      # install project + dev tools (Python 3.10+)
python collect_data.py       # step 1: record training data (opens webcam)
python collect_data.py hello # record only specific signs
python train.py              # step 2: train, writes models/sign_model.keras
python realtime.py           # step 3: live recognition
pytest                       # run tests
ruff check . && ruff format --check .   # lint and format check
```

Camera, GPU and network are not available in CI. Tests must not open a camera
or download models.

## File map

| File | Role |
|---|---|
| `config.py` | Single source of truth: vocabulary (`ACTIONS`), sequence length, paths, thresholds |
| `utils.py` | MediaPipe wrapper (`HandTracker`), feature extraction, drawing helpers |
| `collect_data.py` | Records `(SEQ_LEN, NUM_FEATURES)` arrays to `data/raw/<sign>/<n>.npy` |
| `train.py` | Loads data, augments, trains, evaluates, saves model and `labels.json` |
| `realtime.py` | Live loop: buffer -> predict -> smooth -> sentence -> overlay |
| `architecture.pdf` | Architecture reference |
| `development_plan.md` | Roadmap and task checklist |

## Rules that must not be broken

1. **Feature layout is a contract.** `collect_data.py`, `train.py` and
   `realtime.py` must all use `utils.extract_features()`. If you change the
   features (size, order, normalization), also update `FEATURES_PER_HAND` /
   `NUM_FEATURES` in `config.py`, and tell the user that existing recordings and
   trained models are now invalid and must be regenerated.
2. **Never hard-code vocabulary, paths or thresholds** outside `config.py`.
3. **Keep the public API of `utils.py` stable:** `create_hands`, `detect`,
   `hands_present`, `extract_features`, `draw_hands`, `put_text_box`.
4. **Use the MediaPipe Tasks API** (`HandLandmarker`). `mp.solutions` is removed
   in newer MediaPipe releases; do not reintroduce it.
5. **Augment only the training split** in `train.py`, never the test split.
6. **`idle` must stay in `ACTIONS`.** Without it the model outputs random words
   when nobody is signing.
7. **Do not commit** recorded data, trained models or `hand_landmarker.task`
   (see `.gitignore`). Recordings describe a real person's hands; treat them as
   personal data.
8. Do not add heavy dependencies without a clear need; keep install simple.

## Code style

- Python 3.10+, type hints on new public functions, docstrings on modules and
  non-trivial functions.
- Format and lint with `ruff` (configured in `pyproject.toml`).
- Prefer small pure functions (feature code, smoothing logic) that can be unit
  tested without a camera.
- Keep real-time code fast: avoid per-frame allocations that are not needed,
  call the model with `model(x, training=False)` instead of `model.predict`.

## Testing guidance

- Unit-test feature extraction with fake landmark objects (no camera).
- Unit-test the sentence-building/smoothing logic by feeding scripted label
  sequences. Consider extracting it from `realtime.py` into its own module first.
- Model tests: check input/output shapes and that a few training steps run on
  random data.

## Definition of done for a change

- Code runs end to end (`collect_data` -> `train` -> `realtime`) or the part you
  touched is covered by a test.
- `ruff` passes.
- `README.md`, `development_plan.md` and `architecture.pdf` updated if behaviour
  or structure changed.
- User-visible breaking changes (retrain needed, config renamed) are stated
  clearly in your summary.

## Safety and responsible use

This is a learning/prototype system, not a certified communication aid. Do not
describe it as reliable for real-world communication. Real validation requires
testing with Deaf and hard-of-hearing signers.
