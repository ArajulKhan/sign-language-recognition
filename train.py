"""Train the sequence classifier on collected landmark sequences.

Usage:  python train.py
Outputs: models/sign_model.keras, models/labels.json
"""
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config


def load_dataset():
    X, y, labels = [], [], []
    for action in config.ACTIONS:
        folder = config.DATA_DIR / action
        files = sorted(folder.glob("*.npy")) if folder.exists() else []
        if not files:
            print(f"[warn] no data for '{action}', skipping")
            continue
        labels.append(action)
        idx = len(labels) - 1
        for f in files:
            seq = np.load(f)
            if seq.shape == (config.SEQ_LEN, config.NUM_FEATURES):
                X.append(seq)
                y.append(idx)
    if len(labels) < 2:
        raise SystemExit("Need data for at least 2 actions. Run collect_data.py first.")
    return np.array(X, np.float32), np.array(y), labels


def augment(X, y, copies):
    """Cheap augmentation: gaussian noise, small time shift, small amplitude scaling."""
    rng = np.random.default_rng(42)
    Xa, ya = [X], [y]
    for _ in range(copies):
        noise = rng.normal(0, 0.01, X.shape).astype(np.float32)
        scale = rng.uniform(0.95, 1.05, (X.shape[0], 1, 1)).astype(np.float32)
        shifted = np.roll(X, rng.integers(-3, 4), axis=1)
        Xa.append(shifted * scale + noise)
        ya.append(y)
    return np.concatenate(Xa), np.concatenate(ya)


def build_model(num_classes):
    inp = keras.Input(shape=(config.SEQ_LEN, config.NUM_FEATURES))
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    tf.keras.utils.set_random_seed(42)
    X, y, labels = load_dataset()
    print(f"Loaded {len(X)} sequences, {len(labels)} classes: {labels}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=42)
    X_tr, y_tr = augment(X_tr, y_tr, config.AUGMENT_COPIES)  # augment ONLY the train split

    model = build_model(len(labels))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]
    model.fit(X_tr, y_tr, validation_data=(X_te, y_te),
              epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, callbacks=callbacks)

    # ---- evaluation ----
    probs = model.predict(X_te, verbose=0)
    pred = probs.argmax(axis=1)
    print("\nClassification report:\n", classification_report(y_te, pred, target_names=labels, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_te, pred))

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(config.MODEL_PATH)
    with open(config.LABELS_PATH, "w") as f:
        json.dump(labels, f)
    print("Saved", config.MODEL_PATH)


if __name__ == "__main__":
    main()
