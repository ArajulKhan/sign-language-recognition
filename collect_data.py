"""Record training sequences from webcam."""

import sys
import cv2
import numpy as np

import config
from utils import create_hands, detect, extract_features, draw_hands, put_text_box


def next_index(folder):
    existing = [int(p.stem) for p in folder.glob("*.npy") if p.stem.isdigit()]
    return max(existing) + 1 if existing else 0


def main():
    actions = sys.argv[1:] or config.ACTIONS
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with create_hands() as hands:
        for action in actions:
            folder = config.DATA_DIR / action
            folder.mkdir(parents=True, exist_ok=True)
            start = next_index(folder)

            for seq in range(config.NUM_SEQUENCES):
                # ---- countdown pause: get into position ----
                t0 = cv2.getTickCount()
                while (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000 < config.START_DELAY_MS:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)
                    put_text_box(frame, f"GET READY: '{action}'  ({seq + 1}/{config.NUM_SEQUENCES})",
                                 (15, 40), 0.8, (0, 255, 255))
                    cv2.imshow("Collect", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cap.release(); cv2.destroyAllWindows(); return

                # ---- record SEQ_LEN frames ----
                frames = []
                while len(frames) < config.SEQ_LEN:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)
                    results = detect(frame, hands)
                    frames.append(extract_features(results))
                    draw_hands(frame, results)
                    put_text_box(frame, f"RECORDING '{action}'  frame {len(frames)}/{config.SEQ_LEN}",
                                 (15, 40), 0.8, (0, 255, 0))
                    cv2.imshow("Collect", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cap.release(); cv2.destroyAllWindows(); return

                np.save(folder / f"{start + seq}.npy", np.array(frames, dtype=np.float32))

    cap.release()
    cv2.destroyAllWindows()
    print("Done. Data saved to", config.DATA_DIR)


if __name__ == "__main__":
    main()
