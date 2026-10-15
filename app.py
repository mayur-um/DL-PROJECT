"""Streamlit app for handwritten digit recognition using a trained CNN."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras

MODEL_PATH = Path("digit_model.keras")
ALLOWED_TYPES = ["jpg", "jpeg", "png"]

st.set_page_config(
    page_title="Handwritten Digit Recognition | CNN",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom modern CSS styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 15%, #181c2f 0%, #0d0f1a 50%, #080910 100%);
    color: #e2e8f0;
}

/* Header & Banner */
.hero-container {
    text-align: center;
    padding: 2.2rem 1rem 1.5rem 1rem;
    max-width: 850px;
    margin: 0 auto;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.35);
    color: #a5b4fc;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    margin-bottom: 0.9rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.hero-title {
    font-size: clamp(2.2rem, 4vw, 3.4rem);
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 20%, #cbd5e1 60%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
    margin-bottom: 0.75rem;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    line-height: 1.6;
    margin-bottom: 0.5rem;
}

/* Glassmorphism Cards */
.glass-panel {
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 1.25rem;
    padding: 1.6rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(12px);
    margin-bottom: 1.2rem;
}

.panel-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1.05rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 1rem;
    letter-spacing: 0.02em;
}

/* Prediction highlight card */
.prediction-box {
    background: linear-gradient(145deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.08) 100%);
    border: 1px solid rgba(129, 140, 248, 0.35);
    border-radius: 1.25rem;
    padding: 1.75rem;
    text-align: center;
    box-shadow: 0 15px 35px -10px rgba(99, 102, 241, 0.25);
    margin-bottom: 1.2rem;
}

.digit-display {
    font-size: 5rem;
    font-weight: 800;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #a5b4fc 0%, #c084fc 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.5rem 0;
    filter: drop-shadow(0 0 25px rgba(168, 85, 247, 0.45));
}

.confidence-tag {
    display: inline-block;
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.35);
    color: #4ade80;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 0.35rem 1.1rem;
    border-radius: 999px;
}

.sequence-container {
    display: flex;
    gap: 0.75rem;
    justify-content: center;
    flex-wrap: wrap;
    margin: 1.2rem 0;
}
.sequence-digit {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    width: 3.8rem;
    height: 4.2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 0.85rem;
    box-shadow: 0 8px 20px rgba(79, 70, 229, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.75rem;
    background: rgba(255, 255, 255, 0.04);
    padding: 0.35rem;
    border-radius: 0.9rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0.65rem;
    padding: 0.6rem 1.6rem;
    color: #94a3b8;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4);
}

/* Primary Button Styling */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: white;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.65rem 1.75rem;
    border-radius: 0.75rem;
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 8px 22px rgba(79, 70, 229, 0.35);
    transition: all 0.2s ease;
}
div.stButton > button:first-child:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(79, 70, 229, 0.55);
}
div.stButton > button:first-child:active {
    transform: translateY(0);
}

/* Metric styling */
[data-testid="stMetricValue"] {
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    color: #f8fafc !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.82rem !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Canvas styling */
iframe[title="streamlit_drawable_canvas.streamlit_drawable_canvas"] {
    background-color: #000000 !important;
    border-radius: 16px;
    border: 2px solid rgba(99, 102, 241, 0.45);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7);
    overflow: hidden;
}

#MainMenu, footer, header {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


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


def sort_components_in_reading_order(
    components: list[tuple[int, int, int, int, int]]
) -> list[tuple[int, int, int, int, int]]:
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


def preprocess_canvas_drawing(image_data: np.ndarray) -> np.ndarray:
    """Preprocess drawn canvas RGBA into 28x28 normalized MNIST-format image."""
    if image_data.shape[-1] == 4:
        alpha = image_data[:, :, 3].astype(np.float32) / 255.0
        rgb_max = np.max(image_data[:, :, :3], axis=-1).astype(np.float32)
        gray = rgb_max * alpha
    else:
        gray = np.max(image_data[:, :, :3], axis=-1).astype(np.float32)

    pixel_array = gray.astype(np.uint8)

    # Find bounding box of foreground strokes
    foreground = pixel_array > 25
    if not np.any(foreground):
        return np.zeros((28, 28, 1), dtype=np.float32)

    rows = np.any(foreground, axis=1)
    cols = np.any(foreground, axis=0)
    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    return build_digit_crop(pixel_array, (min_row, min_col, max_row, max_col))


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
    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">⚡ Deep Learning · CNN Classifier</div>
        <h1 class="hero-title">Handwritten Digit Recognition</h1>
        <p class="hero-subtitle">
            Draw any digit directly on your screen or upload an image of single/multi-digit handwritten numbers to classify with deep neural network accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        model = load_model()
    except FileNotFoundError as error:
        st.error(str(error))
        st.info("Train the model first by running: `python train_model.py`")
        return

    # Tabs for Drawing vs Uploading
    tab_draw, tab_upload = st.tabs(["✏️ Draw on Screen", "📁 Upload Image"])

    # --- TAB 1: DRAW ON SCREEN ---
    with tab_draw:
        st.markdown("""
        <div style="margin: 0.5rem 0 1rem 0; color: #94a3b8; font-size: 0.95rem;">
            Draw any single digit from <b>0 to 9</b> on the canvas using your mouse, stylus, or touchscreen.
        </div>
        """, unsafe_allow_html=True)

        col_canvas, col_ctrl = st.columns([1.2, 1], gap="large")

        with col_canvas:
            st.markdown("<div class='panel-header'>🎨 Drawing Canvas</div>", unsafe_allow_html=True)
            stroke_width = st.slider("Brush Thickness", min_value=12, max_value=32, value=22, step=2)
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=stroke_width,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=300,
                width=300,
                drawing_mode="freedraw",
                key="draw_digit_canvas",
                update_streamlit=True,
            )
            st.caption("💡 Center your digit on the canvas for optimal recognition accuracy.")

        with col_ctrl:
            st.markdown("<div class='panel-header'>⚙️ Recognition Control</div>", unsafe_allow_html=True)
            predict_btn = st.button("🔮 Predict Drawn Digit", use_container_width=True)

            has_drawn = canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, :3] > 25)

            if predict_btn or has_drawn:
                if not has_drawn:
                    st.warning("Canvas is empty. Draw a digit first!")
                else:
                    processed_digit = preprocess_canvas_drawing(canvas_result.image_data)
                    batch = np.expand_dims(processed_digit, axis=0)
                    predicted_digit, confidence, probabilities = predict_digit(model, batch)

                    st.markdown(f"""
                    <div class="prediction-box">
                        <div style="color: #a5b4fc; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">
                            Predicted Digit
                        </div>
                        <div class="digit-display">{predicted_digit}</div>
                        <div class="confidence-tag">Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    prob_df = pd.DataFrame({
                        "Digit": [str(i) for i in range(10)],
                        "Probability": probabilities
                    }).set_index("Digit")

                    st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 0.3rem;'>Class Probabilities</div>", unsafe_allow_html=True)
                    st.bar_chart(prob_df, height=160)

        # Bottom visualizer of the 28x28 normalized input if drawn
        if has_drawn:
            with st.expander("🔍 View Preprocessed 28×28 Model Input"):
                p_col1, p_col2 = st.columns([1, 4])
                with p_col1:
                    preprocessed_crop = preprocess_canvas_drawing(canvas_result.image_data)
                    st.image(preprocessed_crop.squeeze(), width=100, caption="28x28 Centered MNIST Input", clamp=True)
                with p_col2:
                    st.markdown("""
                    **MNIST Preprocessing Pipeline:**
                    - Tight bounding box detection of stroke pixels
                    - Aspect-ratio preserving scaling into a 20×20 bounding box
                    - Centering inside a 28×28 normalized grayscale canvas via center-of-mass shift
                    """)

    # --- TAB 2: UPLOAD IMAGE ---
    with tab_upload:
        st.markdown("""
        <div style="margin: 0.5rem 0 1rem 0; color: #94a3b8; font-size: 0.95rem;">
            Upload an image containing one or multiple handwritten digits. The model will automatically segment each digit and read the full sequence.
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=ALLOWED_TYPES,
            accept_multiple_files=False,
            label_visibility="collapsed",
        )

        if uploaded_file is None:
            st.info("Please upload a JPG, JPEG, or PNG image to analyze.")
            return

        try:
            image = Image.open(uploaded_file)
        except UnidentifiedImageError:
            st.error("The uploaded file is not a valid image. Please upload a JPG, JPEG, or PNG file.")
            return
        except OSError:
            st.error("The image could not be opened. Please try another file.")
            return

        col_img, col_pred = st.columns([1, 1.2], gap="large")

        with col_img:
            st.markdown("<div class='panel-header'>🖼️ Uploaded Image</div>", unsafe_allow_html=True)
            st.image(image, caption="Original Upload", use_container_width=True)

        with col_pred:
            with st.spinner("Processing image & detecting digits..."):
                try:
                    digit_images = segment_digits(image)
                except Exception as error:
                    st.error(f"Image preprocessing failed: {error}")
                    return

            if not digit_images:
                st.error("No handwritten digits were found in the uploaded image. Please upload a clearer image.")
                return

            predictions = predict_digits(model, digit_images)
            predicted_digits = [str(prediction[0]) for prediction in predictions]
            sequence = "".join(predicted_digits)
            average_confidence = sum(prediction[1] for prediction in predictions) / len(predictions)

            st.markdown("<div class='panel-header'>✨ Sequence Result</div>", unsafe_allow_html=True)

            sequence_html = "".join(f"<div class='sequence-digit'>{d}</div>" for d in predicted_digits)
            st.markdown(f"<div class='sequence-container'>{sequence_html}</div>", unsafe_allow_html=True)

            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Recognized Sequence", sequence)
            m_col2.metric("Average Confidence", f"{average_confidence:.2f}%")

        # Detailed breakdown of individual digits
        st.write("")
        st.markdown("<div class='panel-header'>📊 Segmented Digits Breakdown</div>", unsafe_allow_html=True)

        cols = st.columns(min(len(predictions), 5))
        for index, (predicted_digit, confidence, probabilities) in enumerate(predictions):
            c_idx = index % len(cols)
            with cols[c_idx]:
                st.markdown(f"""
                <div class="glass-panel" style="padding: 1rem; text-align: center;">
                    <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Digit #{index + 1}</div>
                    <div style="font-size: 2.5rem; font-weight: 800; color: #818cf8; font-family: monospace;">{predicted_digit}</div>
                    <div style="font-size: 0.85rem; font-weight: 600; color: #4ade80;">{confidence:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(digit_images[index].squeeze(), width=80, caption=f"Cropped #{index+1}", clamp=True)
                prob_data = pd.DataFrame({"Probability": probabilities}, index=[str(i) for i in range(10)])
                st.bar_chart(prob_data, height=130)


if __name__ == "__main__":
    main()
