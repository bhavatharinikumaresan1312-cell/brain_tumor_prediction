import os
from io import BytesIO
from datetime import datetime

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import preprocess_input

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle,
    KeepTogether,
)


# ----------------------------------
# App Configuration
# ----------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain Tumor Detection — MRI Classification")

st.write(
    "Upload an MRI image to classify it into glioma, "
    "meningioma, no tumor, or pituitary."
)

MODEL_PATH = "brain_tumor_multiclass_model.h5"

CLASS_LABELS = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

CLASS_EXPLANATIONS = {
    "glioma": (
        "Glioma is a broad model category associated with tumors "
        "arising from glial cells."
    ),
    "meningioma": (
        "Meningioma is a model category associated with tumors arising "
        "from the membranes around the brain and spinal cord."
    ),
    "notumor": (
        "The model selected its 'No tumor' category. This does not prove "
        "that no tumor or other condition is present."
    ),
    "pituitary": (
        "Pituitary is a model category associated with tumors of the "
        "pituitary gland."
    ),
}

# ----------------------------------
# Load Model
# ----------------------------------
@st.cache_resource
def load_model(path):
    return keras.models.load_model(path)


if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model not found: {os.path.abspath(MODEL_PATH)}"
    )
    st.stop()


model = load_model(MODEL_PATH)


# ----------------------------------
# Find Model Input Size
# ----------------------------------
try:
    shape = model.input_shape

    if isinstance(shape, list):
        shape = shape[0]

    if shape[1] is not None and shape[2] is not None:
        TARGET_SIZE = (
            int(shape[1]),
            int(shape[2])
        )
    else:
        TARGET_SIZE = (150, 150)

except (AttributeError, TypeError, IndexError):
    TARGET_SIZE = (150, 150)


# ----------------------------------
# Image Preprocessing
# ----------------------------------
def preprocess_image(image):

    image = image.convert("RGB")

    resampling = getattr(
        Image,
        "Resampling",
        Image
    )

    image = ImageOps.fit(
        image,
        TARGET_SIZE,
        method=resampling.LANCZOS
    )

    array = np.asarray(image).astype("float32")

    array = preprocess_input(array)

    return np.expand_dims(array, axis=0)


# ----------------------------------
# Prediction
# ----------------------------------
def predict(image):

    processed = preprocess_image(image)

    predictions = np.asarray(
        model.predict(
            processed,
            verbose=0
        )
    )

    if (
        predictions.ndim != 2
        or predictions.shape[-1] != len(CLASS_LABELS)
    ):
        raise ValueError(
            "The model output does not match "
            "the four configured classes."
        )

    scores = predictions[0]

    # Convert scores safely to percentages.
    total = float(np.sum(scores))

    if 0.99 <= total <= 1.01:
        percentages = scores * 100
    elif 99 <= total <= 101:
        percentages = scores
    else:
        percentages = (scores / total) * 100

    index = int(np.argmax(percentages))

    return (
        CLASS_LABELS[index],
        float(percentages[index]),
        percentages
    )


# ----------------------------------
# Find a Grad-CAM Layer
# ----------------------------------
def find_last_conv_layer(network):

    # Search nested models first.
    for layer in reversed(network.layers):

        if isinstance(layer, keras.Model):

            try:
                found = find_last_conv_layer(layer)

                if found is not None:
                    return layer, found

            except (ValueError, AttributeError):
                pass

    # Search normal Conv2D layers.
    for layer in reversed(network.layers):

        if isinstance(layer, keras.layers.Conv2D):
            return None, layer

    return None


# ----------------------------------
# Grad-CAM Heatmap
# ----------------------------------
def make_gradcam(image, class_index):

    target = find_last_conv_layer(model)

    if target is None:
        raise ValueError(
            "No Conv2D layer was found. "
            "Grad-CAM needs a compatible "
            "convolutional layer."
        )

    parent_model, conv_layer = target

    # If the convolutional layer is inside a nested model,
    # do not create an invalid heatmap.
    if parent_model is not None:
        raise ValueError(
            f"The last convolutional layer "
            f"({conv_layer.name}) is inside "
            f"the nested model "
            f"({parent_model.name}). "
            f"This model requires a "
            f"model-specific Grad-CAM connection."
        )

    try:
        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[
                conv_layer.output,
                model.output
            ]
        )

        processed = preprocess_image(image)

        with tf.GradientTape() as tape:

            feature_maps, predictions = grad_model(
                processed,
                training=False
            )

            class_score = predictions[:, class_index]

        gradients = tape.gradient(
            class_score,
            feature_maps
        )

        if gradients is None:
            raise ValueError(
                "Gradients are unavailable "
                "for the selected layer."
            )

        weights = tf.reduce_mean(
            gradients,
            axis=(1, 2)
        )

        cam = tf.reduce_sum(
            feature_maps[0]
            * weights[0][
                tf.newaxis,
                tf.newaxis,
                :
            ],
            axis=-1
        )

        cam = tf.nn.relu(cam)

        maximum = tf.reduce_max(cam)

        if float(maximum.numpy()) <= 1e-8:
            raise ValueError(
                "Grad-CAM produced an empty heatmap."
            )

        cam = cam / maximum

        cam = tf.image.resize(
            cam[..., tf.newaxis],
            image.size[::-1]
        ).numpy().squeeze()

        return np.clip(cam, 0, 1)

    except Exception as exc:
        raise ValueError(
            f"Could not generate Grad-CAM: {exc}"
        ) from exc


# ----------------------------------
# Create Grad-CAM Overlay
# ----------------------------------
def create_overlay(image, heatmap):

    original = image.convert("RGB").resize(
        (
            heatmap.shape[1],
            heatmap.shape[0]
        )
    )

    colored = plt.get_cmap("jet")(heatmap)[..., :3]

    colored = (
        colored * 255
    ).astype(np.uint8)

    colored_image = Image.fromarray(colored)

    overlay = Image.blend(
        original,
        colored_image,
        alpha=0.4
    )

    return overlay


# ----------------------------------
# Create Probability Chart Image
# ----------------------------------
def create_probability_chart(percentages):

    fig, ax = plt.subplots(figsize=(10, 4.5))

    x_positions = np.arange(len(CLASS_LABELS))

    ax.bar(
        x_positions,
        percentages
    )

    ax.set_xticks(x_positions)

    ax.set_xticklabels(
        [name.capitalize() for name in CLASS_LABELS]
    )

    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Brain Tumor Classification Probability")
    ax.grid(axis="y", alpha=0.3)

    for i, value in enumerate(percentages):
        ax.text(
            i,
            min(float(value) + 2, 98),
            f"{float(value):.2f}%",
            ha="center"
        )

    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)

    buffer.seek(0)
    return buffer


# ----------------------------------
# Generate Full PDF Report
# ----------------------------------
def generate_pdf_report(
    image,
    label,
    confidence,
    percentages,
    gradcam_overlay=None,
    gradcam_error=None
):

    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=12
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=10,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14
    )

    story = []

    # ----------------------------------
    # Report Header
    # ----------------------------------
    story.append(
        Paragraph(
            "Brain Tumor Detection - MRI Classification Report",
            title_style
        )
    )

    story.append(
        Paragraph(
            f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
            normal_style
        )
    )

    story.append(Spacer(1, 16))

    # ----------------------------------
    # Prediction Result
    # ----------------------------------
    story.append(
        Paragraph("Prediction Result", heading_style)
    )

    result_table = Table(
        [
            ["Predicted Class", label.capitalize()],
            ["Model Score", f"{confidence:.2f}%"]
        ],
        colWidths=[180, 280]
    )

    result_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8EEF7")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("PADDING", (0, 0), (-1, -1), 8),
        ])
    )

    story.append(result_table)
    story.append(Spacer(1, 18))

    # ----------------------------------
    # Uploaded MRI
    # ----------------------------------
    story.append(
        Paragraph("Uploaded MRI Image", heading_style)
    )

    image_buffer = BytesIO()
    image.convert("RGB").save(image_buffer, format="PNG")
    image_buffer.seek(0)

    story.append(
        PDFImage(
            image_buffer,
            width=4.5 * inch,
            height=4.5 * inch
        )
    )

    story.append(Spacer(1, 18))

    # ----------------------------------
    # Classification Scores
    # ----------------------------------
    story.append(
        Paragraph("Classification Scores", heading_style)
    )

    probability_data = [["Class", "Model Score"]]

    for class_name, probability in zip(
        CLASS_LABELS,
        percentages
    ):
        probability_data.append([
            class_name.capitalize(),
            f"{float(probability):.2f}%"
        ])

    probability_table = Table(
        probability_data,
        colWidths=[220, 220]
    )

    probability_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 1), (1, -1), "RIGHT"),
            ("PADDING", (0, 0), (-1, -1), 8),
        ])
    )

    story.append(probability_table)
    story.append(Spacer(1, 16))

    # ----------------------------------
    # Probability Chart
    # ----------------------------------
    story.append(
        Paragraph("Probability Chart", heading_style)
    )

    chart_buffer = create_probability_chart(percentages)

    story.append(
        PDFImage(
            chart_buffer,
            width=6.5 * inch,
            height=3.2 * inch
        )
    )

    story.append(Spacer(1, 12))

    # ----------------------------------
    # Grad-CAM
    # ----------------------------------
    story.append(
        Paragraph("Grad-CAM Explainability", heading_style)
    )

    story.append(
        Paragraph(
            "Grad-CAM highlights image regions that contributed "
            "more strongly to the selected model prediction. "
            "It is an explanation of model behavior and is not "
            "tumor segmentation or a medical diagnosis.",
            normal_style
        )
    )

    story.append(Spacer(1, 10))

    if gradcam_overlay is not None:

        gradcam_buffer = BytesIO()
        gradcam_overlay.convert("RGB").save(
            gradcam_buffer,
            format="PNG"
        )
        gradcam_buffer.seek(0)

        story.append(
            PDFImage(
                gradcam_buffer,
                width=5.5 * inch,
                height=4.0 * inch
            )
        )

    else:
        story.append(
            Paragraph(
                "Grad-CAM was not available for this prediction. "
                "The prediction and classification scores are still "
                "included in this report.",
                normal_style
            )
        )

        if gradcam_error:
            story.append(Spacer(1, 6))
            story.append(
                Paragraph(
                    f"Grad-CAM status: {gradcam_error}",
                    normal_style
                )
            )

    story.append(Spacer(1, 18))

    # ----------------------------------
    # Disclaimer
    # ----------------------------------
    story.append(
        Paragraph(
            "<b>Disclaimer:</b> This report is generated by an "
            "AI-based image classification system. The model score "
            "is not a measure of diagnostic certainty. Grad-CAM is "
            "an explanation of model behavior and is not tumor "
            "segmentation or a medical diagnosis. This report is "
            "intended for project/research demonstration purposes "
            "and should not be used as a substitute for professional "
            "medical evaluation.",
            normal_style
        )
    )

    document.build(story)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


# ----------------------------------
# Upload MRI Image
# ----------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an MRI image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

    except Exception:
        st.error(
            "Unable to read this image. "
            "Please upload a valid MRI image."
        )
        st.stop()

    st.subheader("🖼️ Uploaded MRI")

    preview_image = image.copy()
    preview_image.thumbnail(
        (480, 360),
        getattr(Image, "Resampling", Image).LANCZOS
    )

    st.image(
        preview_image,
        use_container_width=False
    )

    # ----------------------------------
    # Prediction Button
    # ----------------------------------
    if st.button(
        "🔍 Predict and Explain",
        type="primary"
    ):

        with st.spinner("Analyzing MRI..."):

            try:
                label, confidence, percentages = predict(image)

            except Exception as exc:
                st.error(
                    f"Prediction failed: {exc}"
                )
                st.stop()

            st.session_state["result"] = {
                "label": label,
                "confidence": confidence,
                "percentages": percentages,
                "image": image
            }

            st.session_state["gradcam_overlay"] = None
            st.session_state["gradcam_error"] = None

    # ----------------------------------
    # Display Results
    # ----------------------------------
    if "result" in st.session_state:

        result = st.session_state["result"]

        label = result["label"]
        confidence = result["confidence"]
        percentages = result["percentages"]
        result_image = result["image"]

        st.divider()

        st.subheader("🧠 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Class",
                label.capitalize()
            )

        with col2:
            st.metric(
                "Model Score",
                f"{confidence:.2f}%"
            )

        st.caption(
            "The model score represents the model's output for the "
            "predicted class and is not a measure of diagnostic certainty."
        )

        st.subheader("💬 AI Prediction Explanation")
        st.write(
            f"The model selected **{label.capitalize()}** with an output "
            f"score of **{confidence:.2f}%**. "
            f"{CLASS_EXPLANATIONS[label]} "
            "A class label alone does not confirm a tumor or establish "
            "its location."
        )
        st.info(
            "This is an AI-generated result, not a medical diagnosis. "
            "It may be incorrect and should not replace assessment by a "
            "qualified healthcare professional."
        )

        # ----------------------------------
        # Probability Chart
        # ----------------------------------
        st.subheader("📊 Prediction Probability")

        fig, ax = plt.subplots(figsize=(10, 4.5))

        x_positions = np.arange(len(CLASS_LABELS))

        ax.bar(
            x_positions,
            percentages
        )

        ax.set_xticks(x_positions)

        ax.set_xticklabels(
            [name.capitalize() for name in CLASS_LABELS]
        )

        ax.set_ylabel("Probability (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Brain Tumor Classification Probability")
        ax.grid(axis="y", alpha=0.3)

        for i, value in enumerate(percentages):
            ax.text(
                i,
                min(float(value) + 2, 98),
                f"{float(value):.2f}%",
                ha="center"
            )

        fig.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)

        # ----------------------------------
        # Grad-CAM
        # ----------------------------------
        st.subheader("🔥 Grad-CAM Heatmap")

        st.write(
            "The heatmap highlights image regions that contributed "
            "more strongly to the model's selected prediction. It is "
            "an explanation of model behavior, not a tumor segmentation "
            "or medical diagnosis."
        )

        try:

            heatmap = make_gradcam(
                result_image,
                CLASS_LABELS.index(label)
            )

            overlay = create_overlay(
                result_image,
                heatmap
            )

            st.session_state["gradcam_overlay"] = overlay
            st.session_state["gradcam_error"] = None

            original_col, heatmap_col = st.columns(2)

            with original_col:
                st.image(
                    result_image,
                    caption="Original MRI",
                    use_container_width=True
                )

            with heatmap_col:
                st.image(
                    overlay,
                    caption=(
                        "Grad-CAM Overlay "
                        "(hotter regions = stronger influence)"
                    ),
                    use_container_width=True
                )

        except Exception as exc:

            st.session_state["gradcam_overlay"] = None
            st.session_state["gradcam_error"] = str(exc)

            st.warning(
                "Grad-CAM could not be generated "
                "for this model architecture."
            )

            st.code(str(exc))

            st.info(
                "The prediction and probability chart are still available. "
                "The model's layer connections need to be inspected to "
                "adapt Grad-CAM correctly."
            )

        # ----------------------------------
        # PDF Download
        # ----------------------------------
        st.divider()

        st.subheader("📄 Download Full Report")

        st.write(
            "Download the complete prediction report containing the "
            "uploaded MRI, predicted class, model score, probability "
            "chart, classification scores, and Grad-CAM visualization "
            "when available."
        )

        pdf_bytes = generate_pdf_report(
            image=result_image,
            label=label,
            confidence=confidence,
            percentages=percentages,
            gradcam_overlay=st.session_state.get(
                "gradcam_overlay",
                None
            ),
            gradcam_error=st.session_state.get(
                "gradcam_error",
                None
            )
        )

        st.download_button(
            label="📥 Download Full PDF Report",
            data=pdf_bytes,
            file_name="Brain_Tumor_Classification_Report.pdf",
            mime="application/pdf"
        )
