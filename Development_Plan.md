# Development Plan

Goal: a real-time system that recognizes signs from a webcam and shows the
resulting sentence on the video frame.

Status legend: `[x]` done, `[ ]` to do.

## Phase 0: Foundation (done)

- [x] Choose approach: MediaPipe landmarks + sequence model (lighter and more robust than raw pixels)
- [x] Project structure: `config.py`, `utils.py`, `collect_data.py`, `train.py`, `realtime.py`
- [x] MediaPipe Tasks API wrapper (`HandTracker`)
- [x] Feature extraction: wrist-relative, scale-normalized, 2 x 65 values per frame
- [x] Data collection script, training script, real-time script with sentence builder
- [x] Documentation, `.gitignore`, `pyproject.toml`

## Phase 1: First working vocabulary (Week 1)

Target: 7 to 10 signs working live.

- [ ] Choose the sign language (ASL / ISL / BSL ...) and write down the sign for each word
- [ ] Set `ACTIONS` in `config.py` (start small: `idle`, 4 words, optionally `clear`)
- [ ] Record 40 samples per sign (vary speed, distance, angle, lighting)
- [ ] Train; target test accuracy above 90%
- [ ] Run `realtime.py`; tune `CONF_THRESHOLD` and `STABLE_N` until false words are rare

Exit criteria: you can sign a 3-word sentence and it appears correctly 8 of 10 times.

## Phase 2: Data quality (Weeks 2-3)

- [ ] Record with at least 3 different people
- [ ] Split train/test by person, not by random sample
- [ ] Record more `idle` data (hands resting, scratching face, random movement)
- [ ] Grow the vocabulary to 15 to 30 signs
- [ ] Review the confusion matrix and re-record the most confused pairs
- [ ] Add a data-validation script (check shapes, count samples per class, detect empty/zero sequences)

## Phase 3: Model improvements (Weeks 3-5)

- [ ] Add pose and face landmarks (MediaPipe pose/face landmarkers) for sign location and facial expression
- [ ] Add velocity features (frame-to-frame differences)
- [ ] Better augmentation (rotation, mirroring for left-handed signers, time stretching)
- [ ] Compare models: BiLSTM vs 1D-CNN vs small Transformer
- [ ] Track experiments (MLflow or Weights & Biases)

## Phase 4: Real-time quality (Weeks 5-6)

- [ ] Move smoothing and sentence-building logic out of `realtime.py` into a testable module
- [ ] Sign start/end detection (motion-based) instead of a fixed sliding window
- [ ] Export to TFLite or ONNX; measure latency (target under 50 ms per prediction, 30+ FPS)
- [ ] Optional: grammar clean-up of the gloss sequence (rules or a language model)
- [ ] Optional: text-to-speech toggle in the UI

## Phase 5: Engineering and delivery (Weeks 6-8)

- [ ] Unit tests: feature extraction, smoothing, sentence builder, model shapes
- [ ] CI: `ruff` + `pytest` on every push
- [ ] Web UI (FastAPI + WebRTC or Streamlit) or desktop UI
- [ ] Dockerfile and one-command run
- [ ] Final README with demo GIF and measured accuracy/latency

## Phase 6: Future / research

- [ ] Continuous sign recognition (CTC loss or Transformer on public datasets such as WLASL, MS-ASL, How2Sign)
- [ ] Two-way communication: text/speech to sign avatar
- [ ] Mobile app (Android/iOS)
- [ ] User feedback loop for correcting mistakes and retraining
- [ ] Evaluation with Deaf and hard-of-hearing signers

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Model works for the recorder but not others | Multi-person data, person-wise split, augmentation |
| Random words when not signing | Strong `idle` class, confidence threshold, stability window |
| Similar signs confused | More samples, extra features (pose, velocity), check confusion matrix |
| Low FPS | Predict every N frames, TFLite/ONNX, smaller model |
| MediaPipe API changes | Keep all MediaPipe code inside `utils.py` |
| Feature change invalidates data | Treat features as a versioned contract (see `AGENTS.md`) |

## Success metrics

- Test accuracy: above 90% (Phase 1), above 95% (Phase 3) on unseen people
- Live false-word rate: under 1 per minute of idle time
- Latency: under 50 ms per prediction, 20+ FPS
- Vocabulary: 30+ signs
