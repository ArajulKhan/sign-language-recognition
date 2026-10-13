"""Real-time sign recognition: webcam -> MediaPipe -> model -> sentence on the frame.

Keys:  q = quit | c = clear sentence | b = delete last word | s = speak sentence
"""
import json
import time
from collections import deque

import cv2
import numpy as np
from tensorflow import keras

import config
from utils import create_hands, detect, extract_features, hands_present, draw_hands, put_text_box

try:
    import pyttsx3
    tts = pyttsx3.init()
except Exception:  # TTS is optional
    tts = None


def speak(text):
    if tts and text:
        tts.say(text)
        tts.runAndWait()


def main():
    model = keras.models.load_model(config.MODEL_PATH)
    labels = json.load(open(config.LABELS_PATH))
    print("Labels:", labels)

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    buffer = deque(maxlen=config.SEQ_LEN)
    recent = deque(maxlen=config.STABLE_N)
    sentence = []
    last_committed = None
    no_hand_frames = 0
    current_label, current_conf = "idle", 0.0
    frame_idx, prev_t, fps = 0, time.time(), 0.0

    with create_hands() as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            frame_idx += 1

            results = detect(frame, hands)
            buffer.append(extract_features(results))
            no_hand_frames = 0 if hands_present(results) else no_hand_frames + 1

            # ---------- prediction ----------
            if no_hand_frames > config.MAX_NO_HAND:
                current_label, current_conf = "idle", 1.0
            elif len(buffer) == config.SEQ_LEN and frame_idx % config.PREDICT_EVERY == 0:
                x = np.expand_dims(np.array(buffer, dtype=np.float32), 0)
                probs = model(x, training=False).numpy()[0]
                k = int(np.argmax(probs))
                current_label, current_conf = labels[k], float(probs[k])

            # ---------- smoothing + sentence building ----------
            if frame_idx % config.PREDICT_EVERY == 0:
                recent.append(current_label if current_conf >= config.CONF_THRESHOLD else "uncertain")
                if len(recent) == config.STABLE_N and len(set(recent)) == 1:
                    stable = recent[0]
                    if stable == "idle":
                        last_committed = None            # allows repeating the same word later
                    elif stable != "uncertain" and stable != last_committed:
                        if stable == "clear":
                            sentence.clear()
                        else:
                            sentence.append(stable)
                        last_committed = stable

            # ---------- overlay ----------
            draw_hands(frame, results)
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1 / max(now - prev_t, 1e-6))
            prev_t = now

            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h - 70), (w, h), (0, 0, 0), -1)
            text = " ".join(sentence[-config.MAX_WORDS_SHOWN:])
            cv2.putText(frame, text or "...", (12, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            color = (0, 255, 0) if current_conf >= config.CONF_THRESHOLD else (0, 165, 255)
            put_text_box(frame, f"{current_label} {current_conf:.0%}", (12, 35), 0.8, color)
            cv2.putText(frame, f"FPS {fps:.0f}", (w - 110, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            cv2.imshow("Sign Language Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                sentence.clear()
            elif key == ord("b") and sentence:
                sentence.pop()
            elif key == ord("s"):
                speak(" ".join(sentence))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
