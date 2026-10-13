# Sign Language Sentence Recognition (MediaPipe + OpenCV + LSTM)

Webcam -> MediaPipe hand landmarks -> 30-frame sequence -> Conv1D + BiLSTM -> smoothing -> sentence shown on the video frame.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Use Python 3.10 or 3.11. Newer Python versions may not have compatible MediaPipe/TensorFlow wheels.

## Workflow

1. **Choose your vocabulary** in `config.py` (`ACTIONS`). Keep `idle` and `clear`.
2. **Collect data:** `python collect_data.py`
   (or only some signs: `python collect_data.py hello thanks`)
   - 40 samples per sign is the minimum; aim for 60-100.
   - Vary speed, distance, angle, lighting. Record with several people if you can.
   - For `idle`: hands down, hands resting, random non-sign movement.
3. **Train:** `python train.py`
   - Prints a classification report and confusion matrix on a held-out test set.
4. **Run live:** `python realtime.py`
   - `q` quit, `c` clear, `b` delete last word, `s` speak sentence.

## How sentences are built

The model predicts one sign from the last 30 frames. A word is added to the sentence only when:
- confidence >= `CONF_THRESHOLD`, and
- the same prediction repeats `STABLE_N` times in a row, and
- it differs from the last committed word (returning to `idle` lets you repeat a word).

So you sign word by word, briefly lowering your hands between words, and the sentence appears at the bottom of the frame.

## Tuning

| Problem | Fix |
|---|---|
| Words appear by accident | Raise `CONF_THRESHOLD` / `STABLE_N`, record more `idle` data |
| Words are slow to appear | Lower `STABLE_N` |
| Confuses two signs | Record more samples of both, with more variety |
| Works for you but not others | Record other people; split train/test by person |
| Low FPS | Raise `PREDICT_EVERY`, or convert the model to TFLite |

## Known limitations and next steps

- This is **isolated-sign classification + sentence assembly**, not true continuous recognition.
- Only hands are used. Many signs also depend on face, body position and facial expression: switch to `mp.solutions.holistic` (hands + pose + face) and extend `extract_features`.
- Sign language grammar differs from spoken language. The output is a sequence of glosses (e.g. "you name what"). A grammar-fixing step (rules or an LLM) can turn it into natural text.
- For real continuous signing: train with CTC loss or a Transformer on a dataset such as WLASL, MS-ASL or How2Sign.
- Validate with real signers from the Deaf community before presenting it as a communication tool.
