"""Train a CNN on the MNIST dataset and save it as digit_model.keras."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

MODEL_PATH = "digit_model.keras"
EPOCHS = 5
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)


def build_model() -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def print_training_summary(history: keras.callbacks.History) -> None:
    print("\nTraining history")
    for epoch_index in range(len(history.history["loss"])):
        epoch_number = epoch_index + 1
        train_loss = history.history["loss"][epoch_index]
        train_accuracy = history.history["accuracy"][epoch_index]
        val_loss = history.history["val_loss"][epoch_index]
        val_accuracy = history.history["val_accuracy"][epoch_index]
        print(
            f"Epoch {epoch_number}: "
            f"loss={train_loss:.4f}, accuracy={train_accuracy:.4f}, "
            f"val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}"
        )


def main() -> None:
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()

    print("Starting training...")
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1,
    )

    print_training_summary(history)

    print("\nEvaluating on the MNIST test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
