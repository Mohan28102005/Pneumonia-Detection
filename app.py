import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import pydicom
from PIL import Image, ImageDraw
import tensorflow as tf
import os
import cv2

# --- PAGE CONFIG ---
st.set_page_config(page_title="PneumoScan AI", page_icon="🫁", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Dark sidebar */
    [data-testid="stSidebar"] {
        background-color: #1E1E24;
    }
    /* Labels */
    .label-pneumonia {
        color: #ff4b4b;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .label-normal {
        color: #00cc96;
        font-weight: bold;
        font-size: 1.2rem;
    }
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #2b2b36;
        border: 1px solid #454555;
        padding: 10px 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .quote-box {
        background-color: #2b2b36;
        border-left: 3px solid #E8583F;
        padding: 14px 18px;
        border-radius: 4px;
        font-style: italic;
        color: #E7ECF2;
    }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
MODEL_PATHS = {
    "bce": "models/bce_threshold/best_model.keras",
    "densenet": "models/densenet_clinical/best_model.keras"
}

# --- INFERENCE BACKEND ---

@st.cache_resource
def load_model(model_choice):
    """Load model based on choice, cache it so it doesn't reload."""
    model_path = MODEL_PATHS[model_choice]
    if not os.path.exists(model_path):
        return None
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_dicom(dcm_file, image_size):
    try:
        dcm = pydicom.dcmread(dcm_file)
        pixel_array = dcm.pixel_array.astype(np.float32)

        # Apply RescaleSlope/Intercept if present
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            pixel_array = pixel_array * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)

        # Handle MONOCHROME1
        if hasattr(dcm, 'PhotometricInterpretation') and dcm.PhotometricInterpretation == 'MONOCHROME1':
            pixel_array = np.max(pixel_array) - pixel_array

        # Normalize to 0-255 uint8
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-6)
        pixel_array = (pixel_array * 255).astype(np.uint8)

        # Use OpenCV resize exactly like notebook
        img_resized = cv2.resize(pixel_array, (image_size, image_size))

        # Stack to RGB exactly like notebook (no PIL conversion)
        img_rgb = np.stack([img_resized, img_resized, img_resized], axis=-1).astype(np.float32)

        # PIL image just for display (full resolution, no model input)
        pil_image = Image.fromarray(pixel_array)

        # Model input
        input_array = np.expand_dims(img_rgb, axis=0)  # (1, img_size, img_size, 3)

        return pil_image, input_array

    except Exception as e:
        st.error(f"Invalid or corrupted DICOM file: {e}")
        return None, None

def run_inference(model, img_array):
    """Run inference using the loaded Keras model."""
    try:
        preds = model.predict(img_array)

        # Check model output format
        # Assumption: [class_output, bbox_output]
        if isinstance(preds, list) and len(preds) >= 2:
            class_output, bbox_output = preds[0], preds[1]
            prob = float(np.array(class_output).flatten()[0])
            bbox = np.array(bbox_output).flatten()[:4].tolist()
        elif isinstance(preds, np.ndarray):
            st.warning(f"Unexpected output shape: {preds.shape}. Assuming single output classification.")
            # Handling potential shapes like (1, 1) or (1, 2)
            prob = float(np.array(preds).flatten()[0])
            bbox = [0.0, 0.0, 0.0, 0.0]  # Dummy bbox
        else:
            st.warning("Unexpected output type. Returning 0.")
            prob = 0.0
            bbox = [0.0, 0.0, 0.0, 0.0]

        return prob, bbox
    except Exception as e:
        st.error(f"Inference error: {e}")
        return 0.0, [0.0, 0.0, 0.0, 0.0]

def draw_bbox_on_image(pil_image, bbox, probability, threshold):
    img_copy = pil_image.convert('RGB')  # convert here, not earlier
    if probability > threshold:
        draw = ImageDraw.Draw(img_copy)
        w, h = img_copy.size

        x_norm, y_norm, w_norm, h_norm = bbox

        x0 = int(x_norm * w)
        y0 = int(y_norm * h)
        x1 = x0 + int(w_norm * w)
        y1 = y0 + int(h_norm * h)

        x0 = max(0, min(x0, w))
        y0 = max(0, min(y0, h))
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))

        if x1 > x0 and y1 > y0:
            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
            label_text = f"Pneumonia {probability:.1%}"
            draw.text((x0, max(0, y0 - 15)), label_text, fill="red")

    return img_copy

# --- SIDEBAR ---
with st.sidebar:
    st.title("🫁 PneumoScan AI")
    st.markdown("""
    A dual-output deep learning model that classifies pneumonia and localizes
    it on the X-ray, trained on the RSNA Pneumonia Detection Challenge dataset.
    """)
    st.markdown("---")
    st.metric("Patients", "26,684")
    st.metric("Val AUC", "0.84")
    st.metric("Clinical Recall", "0.93")

# --- TABS ---
tabs = st.tabs([
    "01 · Data",
    "02 · Architecture",
    "03 · Loss Functions",
    "04 · Training",
    "05 · Evaluation",
    "Live Inference",
])
tab_data, tab_arch, tab_loss, tab_train, tab_eval, tab_infer = tabs

# ============================================================
# 02 · DATA
# ============================================================
with tab_data:
    st.header("Problem & dataset")
    st.markdown(
        "Pneumonia is a lung infection, diagnosed from chest X-rays. The model has two "
        "jobs: **classify** (pneumonia / normal) and **localize** (draw the box)."
    )

    st.subheader("Dataset — RSNA Pneumonia Detection Challenge")
    dataset_df = pd.DataFrame({
        "Metric": ["Total patients", "Pneumonia", "Normal", "Split", "Source format"],
        "Value": [
            "~26,684",
            "~9,000 (33%)",
            "~17,684 (67%)",
            "85% train / 15% val, stratified by patient",
            "DICOM (.dcm) — the standard medical imaging file format",
        ],
    })
    st.table(dataset_df.set_index("Metric"))

    st.subheader("How the labels were built")
    st.markdown("""
- Some patients have **multiple** bounding boxes (more than one infected region).
- These get merged into a single box using min/max coordinates across all boxes for that
  patient — simplifies the target, at the cost of losing "multiple distinct regions"
  information (a fair limitation to mention if asked).
- Box coordinates are normalized to **[0, 1]** by dividing by the original image size
  (1024px), so the model predicts scale-independent coordinates.
    """)

    st.subheader("Why 33% / 67% matters")
    st.info(
        "That's a real but moderate class imbalance (roughly 2:1). Not extreme, but "
        "enough that a naive model could get decent accuracy by mostly predicting "
        "\"normal\" — which is exactly the failure mode you don't want in a screening "
        "tool. This is why so much of the project's design (loss functions, "
        "oversampling) is built around handling that imbalance in favor of catching "
        "positives."
    )

# ============================================================
# 03 · ARCHITECTURE
# ============================================================
with tab_arch:
    st.header("Model architecture")
    st.markdown(
        "One shared backbone, two task-specific \"heads\" branching off the top. Same "
        "structure was reused across all three experiments — only the backbone and head "
        "sizes changed."
    )

    st.code(
        "Input image (H × W × 3)\n"
        "        ↓\n"
        "Backbone (EfficientNet or DenseNet121, pretrained on ImageNet)\n"
        "        ↓\n"
        "GlobalAveragePooling → BatchNorm → Dense(shared) → Dropout\n"
        "        ↓ splits into two heads ↓\n"
        "Classification head              Localization head\n"
        "Dense → Dropout → Dense(1, sigmoid)   Dense → Dropout → Dense(4, sigmoid)\n"
        "→ pneumonia probability                → [x, y, w, h] box",
        language=None,
    )

    st.subheader("Plain-English explanation of each piece")
    with st.expander("Backbone"):
        st.write(
            "A network pretrained on millions of natural images (ImageNet). It already "
            "knows how to detect edges, textures, and shapes, so you're reusing that "
            "instead of learning from scratch. This is **transfer learning**."
        )
    with st.expander("GlobalAveragePooling"):
        st.write(
            "Takes the backbone's grid of feature maps and squashes each one down to a "
            "single number, giving a compact summary vector instead of a huge spatial grid."
        )
    with st.expander("Shared Dense layer"):
        st.write(
            "One common \"representation\" both heads branch off from, so the model "
            "learns features useful for both tasks at once, not two separate models "
            "bolted together."
        )
    with st.expander("Dropout"):
        st.write(
            "Randomly zeroes out some neurons during training so the model doesn't "
            "over-rely on any single feature. Standard overfitting guard."
        )
    with st.expander("Two heads"):
        st.write(
            "One outputs a single sigmoid number (probability of pneumonia), the other "
            "outputs four sigmoid numbers (the box coordinates, each squashed to 0–1 "
            "since coordinates were normalized)."
        )
    with st.expander("float32 cast on outputs"):
        st.write(
            "Training used mixed precision (float16) for speed and memory, but the "
            "final outputs are cast back to float32 because float16 is numerically "
            "unstable for the last layer's small probability/coordinate values."
        )

    st.subheader("Why dual-output instead of two separate models?")
    st.info(
        "A single network trained on both tasks jointly learns features useful for "
        "both. The localization head acts almost like a regularizer — it forces the "
        "shared features to actually attend to the lung region that matters, rather "
        "than the classification head latching onto spurious shortcuts (image "
        "artifacts, positioning, etc.) that happen to correlate with the label."
    )

# ============================================================
# 04 · LOSS FUNCTIONS
# ============================================================
with tab_loss:
    st.header("Loss functions — three experiments, one evolving goal")
    st.markdown(
        "Every experiment changed the backbone **and** the loss function together, "
        "chasing the same clinical goal: catch as many real pneumonia cases as possible."
    )

    with st.expander("Experiment 1 — EfficientNetB0 + Focal Loss", expanded=True):
        st.markdown("**Goal:** establish a baseline with a light backbone and an imbalance-aware loss.")
        st.markdown(
            "**Focal Loss** (γ=2, α=0.6) modifies plain cross-entropy so that examples "
            "the model already gets right contribute almost nothing to the loss, and "
            "hard/misclassified examples dominate training. That stops the easy "
            "\"normal\" majority class from drowning out learning signal from the "
            "harder pneumonia cases."
        )
        st.code(
            "alpha_t = y_true * alpha + (1-y_true) * (1-alpha)\n"
            "loss    = alpha_t * (1 - p_t)^gamma * bce",
            language="python",
        )

    with st.expander("Experiment 2 — EfficientNetB4 + BCE with label smoothing"):
        st.markdown("**Goal:** scale up to a bigger backbone, and improve calibration.")
        st.markdown("""
- **Label smoothing (0.1)** — instead of forcing the model to predict exactly 0 or 1, targets become 0.05 / 0.95. This stops the model from becoming overconfident and generalizes slightly better.
- **Huber loss** for the box coordinates instead of MSE — behaves like L1 (steady, small gradient) for large errors and L2 (smooth) for small ones, so a few badly-labeled or outlier boxes don't blow up training.
- Added a custom **F1 metric** tracked live during training for better monitoring than accuracy alone.
        """)
        st.success(
            "Result: val AUC climbed to ~0.84. Val recall consistently beat val "
            "precision (~0.58–0.67 vs ~0.40–0.45) — the model was already leaning "
            "toward catching positives."
        )

    with st.expander("Experiment 3 — the clinical run: DenseNet121 + RecallFocused Loss", expanded=True):
        st.markdown(
            "**Goal:** explicitly maximize recall. In screening, a missed pneumonia "
            "case (false negative) is far more dangerous than a false alarm."
        )
        st.markdown(
            "**RecallFocused Loss** — a custom asymmetric binary cross-entropy that "
            "weights false negatives higher than false positives:"
        )
        st.code(
            "pos_loss = -y_true * log(y_pred)\n"
            "neg_loss = -(1-y_true) * log(1-y_pred)\n"
            "loss     = fn_weight * pos_loss + neg_loss   # fn_weight starts at 3.0",
            language="python",
        )
        st.markdown(
            "**fn_weight went from 3.0 → 2.0 mid-training** — after watching recall "
            "shoot up but precision collapse too far, the model was recompiled with a "
            "softer weight and training resumed from the existing checkpoint. This is "
            "a great story to tell in an interview: it shows you monitor training, "
            "notice a trade-off going too far, and correct it rather than blindly "
            "running a fixed recipe."
        )
        st.markdown(
            "Also moved to 512×512 input resolution (up from 160×160) for finer "
            "spatial detail, which forced batch size down to 4 to fit the 4GB GPU."
        )

# ============================================================
# 05 · TRAINING
# ============================================================
with tab_train:
    st.header("Training strategy")

    st.subheader("Two-phase transfer learning (used in all 3 experiments)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Phase 1 — warmup**")
        st.markdown("""
- Backbone fully frozen
- Only the two heads train
- lr = 1e-4, 10 epochs
        """)
    with c2:
        st.markdown("**Phase 2 — fine-tune**")
        st.markdown("""
- Top backbone layers unfrozen
- Lower learning rate
- lr = 1e-5, 30–50 epochs
        """)

    st.markdown("**Why freeze first?**")
    st.info(
        "At the very start, the heads are randomly initialized and produce large, "
        "noisy gradients. If the backbone were trainable from step one, those noisy "
        "gradients would flow back and wreck the good pretrained features. Freezing "
        "protects the backbone until the heads have learned something sensible — only "
        "then do you carefully unfreeze the top (most task-specific) layers and "
        "fine-tune with a small learning rate. Earlier backbone layers (edges, "
        "textures) stay frozen throughout since those low-level features are useful "
        "for any image task."
    )

    st.subheader("Hardware-driven decisions (RTX 3050, 4GB VRAM)")
    st.markdown("""
- **Mixed precision (float16)** — roughly halves memory use and speeds up training on modern GPUs, with negligible accuracy loss. Model outputs still cast to float32 for numerical stability.
- **Memory growth** enabled so TensorFlow doesn't grab all GPU memory upfront.
- **Batch size 4** in the 512×512 experiment — the resolution bump ate the memory savings mixed precision bought back.
    """)

    st.subheader("Data pipeline")
    st.code(
        "DICOM file (.dcm) → pixel array\n"
        "        ↓\n"
        "Min-max normalize to [0, 255] → stack grayscale into 3 channels (RGB)\n"
        "        ↓\n"
        "Albumentations augmentation (with BboxParams to keep box + image transforms in sync)\n"
        "        ↓\n"
        "tf.data: shuffle → batch → prefetch",
        language=None,
    )
    st.markdown(
        "DICOM images are natively single-channel (grayscale). ImageNet-pretrained "
        "backbones expect 3-channel input, so the same grayscale array is copied "
        "across all 3 channels rather than discarding the pretrained weights' input "
        "structure."
    )
    st.markdown("**Why BboxParams matters:** if you flip or rotate the image but don't "
                "apply the same transform to the box coordinates, your labels become "
                "wrong. Albumentations' BboxParams handles that synchronization "
                "automatically.")

    st.subheader("Handling the class imbalance")
    st.caption(
        "Three different oversampling ratios were tried across the experiments — "
        "repeating pneumonia-positive samples so the model sees them more often "
        "during training:"
    )
    oversampling_df = pd.DataFrame({
        "Experiment": ["Exp 1", "Exp 2", "Exp 3"],
        "Strategy": [
            "Strict 1:1 (positives repeated to match negatives)",
            "1:2 (positives repeated to half the negatives — reduces repetition noise)",
            "1:1 with in-pipeline shuffling",
        ],
    })
    st.table(oversampling_df.set_index("Experiment"))

# ============================================================
# 06 · EVALUATION
# ============================================================
with tab_eval:
    st.header("Evaluation & threshold tuning")
    st.markdown(
        "A sigmoid outputs a probability, not a decision. Picking the cutoff "
        "(threshold) that turns \"0.42\" into \"pneumonia\" or \"not pneumonia\" is "
        "its own optimization step."
    )

    st.subheader("Threshold sweep")
    st.markdown(
        "After training, predictions are swept across thresholds from 0.20 to "
        "0.70/0.75 in small steps, computing precision/recall/F1 at each, and picking "
        "the threshold that maximizes F1 on the validation set."
    )

    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60]
    precision = [0.312, 0.371, 0.421, 0.468, 0.512]
    recall = [0.821, 0.741, 0.643, 0.573, 0.481]
    f1_scores = [0.452, 0.495, 0.510, 0.515, 0.496]

    sweep_df = pd.DataFrame({
        "Threshold": thresholds,
        "Precision": precision,
        "Recall": recall,
        "F1": f1_scores,
    })
    st.table(sweep_df.set_index("Threshold"))
    st.caption("F1 peaks around threshold ≈ 0.40–0.50 (marked ← in the write-up).")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=precision, mode='lines+markers', name='Precision'))
    fig.add_trace(go.Scatter(x=thresholds, y=recall, mode='lines+markers', name='Recall'))
    fig.add_trace(go.Scatter(x=thresholds, y=f1_scores, mode='lines+markers', name='F1 Score'))
    fig.update_layout(title="Threshold Sweep Metrics", xaxis_title="Threshold",
                       yaxis_title="Score", hovermode="x unified", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Lower thresholds trade precision for recall — useful to show off if asked "
        "\"how would you tune this for a stricter screening use case?\" (answer: pick "
        "a lower threshold, accept more false positives, catch more true cases)."
    )

    st.subheader("Sanity checks")
    st.markdown("""
- Logged prediction distribution stats (mean, std, min, max, % of predictions above threshold) to catch **model collapse** — e.g. the model just predicting "pneumonia" for everyone to game recall.
- Tracked AUC, accuracy, precision, and recall throughout training, not just final numbers, to see the trajectory.
    """)

    st.subheader("Inference (single X-ray → prediction)")
    st.code(
        "Load .dcm → normalize + resize to model's input size\n"
        "        ↓\n"
        "model.predict() → [class probability, box coordinates]\n"
        "        ↓\n"
        "If prob ≥ threshold → draw red box, label \"PNEUMONIA DETECTED (confidence: x)\"",
        language=None,
    )

    st.subheader("Headline results to quote")
    c1, c2 = st.columns(2)
    c1.metric("Val AUC", "0.84", "climbed from ~0.75 in warmup")
    c2.metric("Clinical Recall", "0.93", "recall-optimized configuration")
    st.caption(
        "Lead with clinical recall when asked \"what was your best result,\" since it "
        "directly reflects the project's actual goal."
    )

# ============================================================
# LIVE INFERENCE
# ============================================================
with tab_infer:
    st.header("Live Inference Engine")

    active_model_key = "bce"
    img_size = 160

    st.info("Using model: BCE + Threshold (EfficientNetB0, 160px)")
    st.info("Upload a DICOM file to test the model.")
    uploaded_file = st.file_uploader("Upload Chest X-Ray (.dcm)", type=["dcm"])

    # Store in session state
    if "last_prob" not in st.session_state:
        st.session_state.last_prob = 0.0
    if "last_bbox" not in st.session_state:
        st.session_state.last_bbox = [0.0, 0.0, 0.0, 0.0]
    if "last_image" not in st.session_state:
        st.session_state.last_image = None
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = ""

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.uploaded_filename:
            # New file uploaded, run inference
            st.session_state.uploaded_filename = uploaded_file.name

            with st.spinner("Processing DICOM..."):
                pil_image, model_input = preprocess_dicom(uploaded_file, img_size)

            if pil_image and model_input is not None:
                st.session_state.last_image = pil_image

                with st.spinner("Running Inference..."):
                    model = load_model(active_model_key)
                    if model is not None:
                        prob, bbox = run_inference(model, model_input)
                        st.session_state.last_prob = prob
                        st.session_state.last_bbox = bbox
                    else:
                        st.error(f"Please place the model at `{MODEL_PATHS[active_model_key]}` to run inference.")
                        st.session_state.last_prob = 0.0
                        st.session_state.last_bbox = [0.0, 0.0, 0.0, 0.0]

    # Display Results if we have an image
    if st.session_state.last_image is not None:
        st.markdown("---")

        col1, col2 = st.columns([2, 1])

        with col2:
            st.subheader("Inference Results")
            threshold = st.slider("Prediction Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)

            prob = st.session_state.last_prob
            bbox = st.session_state.last_bbox

            # Confidence logic
            distance = abs(prob - threshold)
            if distance > 0.3:
                confidence = "HIGH"
            elif distance > 0.1:
                confidence = "MODERATE"
            else:
                confidence = "LOW"

            is_pneumonia = prob > threshold
            pred_text = "PNEUMONIA" if is_pneumonia else "NORMAL"
            pred_class = "label-pneumonia" if is_pneumonia else "label-normal"

            # Metrics
            st.markdown(f"**Prediction:** <span class='{pred_class}'>{pred_text}</span>", unsafe_allow_html=True)
            st.metric("Pneumonia Probability", f"{prob:.1%}")
            st.metric("Confidence", confidence)

            st.progress(prob, text="Probability Score")

            if is_pneumonia and sum(bbox) > 0:
                st.info(f"Bounding Box (Normalized):\n`[{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]`\n\nThe bounding box highlights the localized area of suspected pneumonia.")

        with col1:
            st.subheader("Chest X-Ray")
            disp_img = draw_bbox_on_image(st.session_state.last_image, bbox, prob, threshold)
            st.image(disp_img, use_container_width=True)
