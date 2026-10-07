"""Streamlit app for handwritten digit recognition using a trained CNN."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from tensorflow import keras

MODEL_PATH = Path("digit_model.keras")
ALLOWED_TYPES = ["jpg", "jpeg", "png"]


st.set_page_config(
    page_title="Handwritten Digit Recognition Using CNN",
    page_icon="0",
    layout="centered",
)


@st.cache_resource
def load_model() -> keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "digit_model.keras not found. Run train_model.py first to train and save the model."
        )
    return keras.models.load_model(MODEL_PATH)


def otsu_threshold(pixel_array: np.ndarray) -> int:
    histogram = np.bincount(pixel_array.ravel(), minlength=256)
    total_pixels = pixel_array.size
    sum_total = float(np.dot(np.arange(256), histogram))

    sum_background = 0.0
    weight_background = 0.0
    best_threshold = 127
    max_variance = -1.0

    for threshold in range(256):
        weight_background += histogram[threshold]
        if weight_background == 0:
            continue

        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_background += threshold * histogram[threshold]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance_between_classes = (
            weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        )

        if variance_between_classes > max_variance:
            max_variance = variance_between_classes
            best_threshold = threshold

    return best_threshold


def connected_components(foreground_mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    visited = np.zeros_like(foreground_mask, dtype=bool)
    height, width = foreground_mask.shape
    components: list[tuple[int, int, int, int, int]] = []

    for row_index in range(height):
        for column_index in range(width):
            if not foreground_mask[row_index, column_index] or visited[row_index, column_index]:
                continue

            stack = [(row_index, column_index)]
            visited[row_index, column_index] = True
            min_row = max_row = row_index
            min_col = max_col = column_index
            area = 0

            while stack:
                current_row, current_col = stack.pop()
                area += 1
                min_row = min(min_row, current_row)
                max_row = max(max_row, current_row)
                min_col = min(min_col, current_col)
                max_col = max(max_col, current_col)

                for row_delta in (-1, 0, 1):
                    for col_delta in (-1, 0, 1):
                        if row_delta == 0 and col_delta == 0:
                            continue

                        next_row = current_row + row_delta
                        next_col = current_col + col_delta
                        if not (0 <= next_row < height and 0 <= next_col < width):
                            continue

                        if foreground_mask[next_row, next_col] and not visited[next_row, next_col]:
                            visited[next_row, next_col] = True
                            stack.append((next_row, next_col))

            components.append((min_row, min_col, max_row, max_col, area))

    return components


def sort_components_in_reading_order(components: list[tuple[int, int, int, int, int]]) -> list[tuple[int, int, int, int, int]]:
    if not components:
        return []

    ordered_components = sorted(components, key=lambda component: (component[0] + component[2]) / 2)
    heights = [component[2] - component[0] + 1 for component in ordered_components]
    median_height = float(np.median(heights)) if heights else 0.0
    line_gap = max(10.0, median_height * 0.6)

    lines: list[list[tuple[int, int, int, int, int]]] = []
    current_line: list[tuple[int, int, int, int, int]] = [ordered_components[0]]
    current_center = (ordered_components[0][0] + ordered_components[0][2]) / 2

    for component in ordered_components[1:]:
        component_center = (component[0] + component[2]) / 2
        if abs(component_center - current_center) <= line_gap:
            current_line.append(component)
            current_center = sum((item[0] + item[2]) / 2 for item in current_line) / len(current_line)
        else:
            lines.append(sorted(current_line, key=lambda item: item[1]))
            current_line = [component]
            current_center = component_center

    lines.append(sorted(current_line, key=lambda item: item[1]))

    reading_order: list[tuple[int, int, int, int, int]] = []
    for line in lines:
        reading_order.extend(line)

    return reading_order


def build_digit_crop(image_array: np.ndarray, bounding_box: tuple[int, int, int, int]) -> np.ndarray:
    min_row, min_col, max_row, max_col = bounding_box
    crop = image_array[min_row : max_row + 1, min_col : max_col + 1]

    crop_height, crop_width = crop.shape
    if crop_height == 0 or crop_width == 0:
        return np.zeros((28, 28, 1), dtype=np.float32)

    scale = 20.0 / max(crop_height, crop_width)
    resized_height = max(1, int(round(crop_height * scale)))
    resized_width = max(1, int(round(crop_width * scale)))
    resized_crop = Image.fromarray(crop).resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    resized_array = np.array(resized_crop, dtype=np.uint8)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    row_offset = (28 - resized_height) // 2
    col_offset = (28 - resized_width) // 2
    canvas[row_offset : row_offset + resized_height, col_offset : col_offset + resized_width] = resized_array

    non_zero_pixels = np.argwhere(canvas > 0)
    if non_zero_pixels.size > 0:
        center_row, center_col = non_zero_pixels.mean(axis=0)
        shift_row = int(round(14 - center_row))
        shift_col = int(round(14 - center_col))
        canvas = np.roll(canvas, shift_row, axis=0)
        canvas = np.roll(canvas, shift_col, axis=1)

    digit_array = canvas.astype("float32") / 255.0
    digit_array = np.expand_dims(digit_array, axis=-1)
    return digit_array


def segment_digits(uploaded_image: Image.Image) -> list[np.ndarray]:
    grayscale_image = ImageOps.autocontrast(uploaded_image.convert("L"))
    pixel_array = np.array(grayscale_image)

    threshold = otsu_threshold(pixel_array)
    dark_pixels = pixel_array <= threshold
    light_pixels = pixel_array > threshold

    if dark_pixels.sum() <= light_pixels.sum():
        foreground_mask = dark_pixels
        model_image = 255 - pixel_array
    else:
        foreground_mask = light_pixels
        model_image = pixel_array

    components = connected_components(foreground_mask)
    min_area = max(25, int(0.0005 * pixel_array.size))
    filtered_components = [component for component in components if component[4] >= min_area]
    filtered_components = sort_components_in_reading_order(filtered_components)

    return [build_digit_crop(model_image, component[:4]) for component in filtered_components]


def predict_digit(model: keras.Model, processed_image: np.ndarray) -> tuple[int, float, np.ndarray]:
    probabilities = model.predict(processed_image, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities)) * 100.0
    return predicted_digit, confidence, probabilities


def predict_digits(model: keras.Model, digit_images: Iterable[np.ndarray]) -> list[tuple[int, float, np.ndarray]]:
    predictions: list[tuple[int, float, np.ndarray]] = []
    for digit_image in digit_images:
        processed_image = np.expand_dims(digit_image, axis=0)
        predictions.append(predict_digit(model, processed_image))
    return predictions


def main() -> None:
    st.title("Handwritten Digit Recognition Using CNN")
    st.write(
        "Upload a JPG, JPEG, or PNG image containing one or more handwritten digits. The app will split the image into separate digits, classify each one, and show the full sequence."
    )

    try:
        model = load_model()
    except FileNotFoundError as error:
        st.error(str(error))
        st.info("Train the model first by running: python train_model.py")
        return

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=ALLOWED_TYPES,
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Please upload an image to get a prediction.")
        return

    try:
        image = Image.open(uploaded_file)
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a JPG, JPEG, or PNG file.")
        return
    except OSError:
        st.error("The image could not be opened. Please try another file.")
        return

    st.subheader("Uploaded Image")
    st.image(image, caption="Uploaded handwritten digits", use_container_width=True)

    try:
        digit_images = segment_digits(image)
    except Exception as error:  # pragma: no cover - defensive user-facing handling
        st.error(f"Image preprocessing failed: {error}")
        return

    if not digit_images:
        st.error("No handwritten digits were found in the uploaded image. Please upload a clearer image.")
        return

    predictions = predict_digits(model, digit_images)
    predicted_digits = [str(prediction[0]) for prediction in predictions]
    sequence = "".join(predicted_digits)
    average_confidence = sum(prediction[1] for prediction in predictions) / len(predictions)

    col1, col2 = st.columns(2)
    col1.metric("Predicted Digits", sequence)
    col2.metric("Average Confidence", f"{average_confidence:.2f}%")

    st.subheader("Prediction Details")
    for index, (predicted_digit, confidence, probabilities) in enumerate(predictions, start=1):
        st.write(f"Digit {index}: **{predicted_digit}** with **{confidence:.2f}%** confidence")
        st.bar_chart(probabilities)

    st.success(
        f"The model predicts the digit sequence {sequence} with an average confidence of {average_confidence:.2f}%."
    )


if __name__ == "__main__":
    main()
