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
st.set_page_config(page_title="PneumoScan AI", page_icon="D", layout="wide")

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
INFERENCE_THRESHOLD = 0.50

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

def render_data_flow(nodes, accent="#55d6be"):
    """Render a compact, responsive vertical process diagram."""
    items = []
    for index, node in enumerate(nodes):
        items.append(
            f"<div class='data-flow-node'><div class='data-flow-index'>{index + 1:02d}</div>"
            f"<div><strong>{node[0]}</strong><span>{node[1]}</span></div></div>"
        )
        if index < len(nodes) - 1:
            items.append(f"<div class='data-flow-arrow' style='color:{accent}'>↓</div>")
    st.markdown("<div class='data-flow'>" + "".join(items) + "</div>", unsafe_allow_html=True)

def render_problem_illustration():
    """Render an explicitly illustrative dual-output chest X-ray concept."""
    st.markdown(
        """
        <div class="problem-illustration">
          <div class="illustration-stage">
            <div class="xray-art" role="img" aria-label="Illustrative chest X-ray with suspected region">
              <div class="lung lung-left"></div><div class="lung lung-right"></div>
              <div class="spine"></div><div class="rib rib-1"></div><div class="rib rib-2"></div>
              <div class="rib rib-3"></div><div class="rib rib-4"></div>
              <div class="suspected-region"></div>
            </div>
            <span class="illustration-label">Illustrative X-ray<br><small>not a model prediction</small></span>
          </div>
          <div class="illustration-arrow">→</div>
          <div class="model-stage"><div class="model-icon">AI</div><strong>Deep learning<br>model</strong><small>shared visual features</small></div>
          <div class="illustration-arrow">→</div>
          <div class="output-stack">
            <div class="output-card classification"><span>01</span><strong>Classification</strong><small>Pneumonia / Normal</small></div>
            <div class="output-card localization"><span>02</span><strong>Localization</strong><small>Suspected region + box</small></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_dataset_structure():
    st.markdown(
        """
        <div class="structure-diagram">
          <div class="structure-root">RSNA Pneumonia Detection Challenge</div>
          <div class="structure-branches">
            <div class="structure-branch"><b>DICOM image files</b><span>Chest X-ray pixel data<br>+ imaging metadata</span></div>
            <div class="structure-branch"><b>Annotation CSV</b><span>Patient ID → target label<br>+ x, y, width, height</span></div>
          </div>
          <div class="structure-link">Patient ID connects one image record to its ground truth</div>
          <div class="structure-record"><span>Patient ID</span><span>Target: 0 / 1</span><span>Box: x, y, w, h</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_data_styles():
    st.markdown(
        """
        <style>
        .section-kicker{color:#55d6be;text-transform:uppercase;letter-spacing:.14em;font-size:.72rem;font-weight:700;margin-bottom:.3rem}
        .data-hero{background:linear-gradient(135deg,#202632 0%,#172b35 100%);border:1px solid #34505a;border-radius:16px;padding:24px 26px;margin:6px 0 20px}
        .data-hero h2{margin:0 0 8px;color:#f4f7fa}.data-hero p{margin:0;color:#bac8d1;max-width:900px;line-height:1.55}
        .problem-illustration{display:flex;align-items:center;justify-content:center;gap:18px;background:#171b22;border:1px solid #303a45;border-radius:14px;padding:22px;min-height:250px}
        .illustration-stage{position:relative;text-align:center}.illustration-label{display:block;color:#9eabb5;font-size:.75rem;margin-top:8px}.illustration-label small{color:#e8a96a}
        .xray-art{position:relative;width:158px;height:176px;border-radius:48% 48% 43% 43%;background:radial-gradient(ellipse at center,#a8c0c5 0,#55717b 34%,#253741 72%,#151c22 100%);overflow:hidden;box-shadow:0 0 20px #719ca966}
        .lung{position:absolute;top:31px;width:52px;height:115px;border-radius:48% 48% 44% 44%;background:radial-gradient(ellipse at 42% 42%,#c9dce0 0,#7999a0 58%,#36505a 100%);opacity:.88}.lung-left{left:23px;transform:rotate(5deg)}.lung-right{right:23px;transform:rotate(-5deg)}
        .spine{position:absolute;left:77px;top:25px;height:130px;width:5px;border-radius:5px;background:#d9e8e8aa}.rib{position:absolute;left:32px;width:94px;height:32px;border-top:2px solid #d5e4e355;border-radius:50%}.rib-1{top:43px}.rib-2{top:68px}.rib-3{top:93px}.rib-4{top:118px}
        .suspected-region{position:absolute;left:37px;top:82px;width:42px;height:31px;border:3px solid #ff806b;border-radius:8px;box-shadow:0 0 12px #ff806b;background:#ff806b22}
        .illustration-arrow{font-size:2rem;color:#55d6be}.model-stage{display:flex;flex-direction:column;align-items:center;text-align:center;color:#e9f2f4;gap:5px}.model-stage small,.output-card small{color:#9eabb5;font-size:.72rem}.model-icon{display:grid;place-items:center;width:64px;height:64px;border-radius:14px;background:#315a63;color:#70f0d7;font-weight:800;border:1px solid #55d6be;font-size:1.25rem}
        .output-stack{display:grid;gap:10px;min-width:175px}.output-card{border-radius:10px;padding:11px 13px;border-left:4px solid;text-align:left;background:#222933}.output-card span{font-size:.68rem;color:#8da0aa;display:block}.output-card strong{display:block;color:#f4f7fa}.classification{border-color:#55d6be}.localization{border-color:#ff806b}
        .metric-card{background:#202632;border:1px solid #35424e;border-radius:12px;padding:14px 16px;height:100%}.metric-card .metric-label{color:#9eabb5;font-size:.78rem}.metric-card .metric-value{font-size:1.45rem;font-weight:700;color:#f4f7fa;margin:4px 0}.metric-card .metric-note{font-size:.75rem;color:#55d6be}
        .data-flow{display:flex;flex-direction:column;align-items:stretch;background:#171b22;border:1px solid #303a45;border-radius:14px;padding:14px 18px}.data-flow-node{display:flex;align-items:center;gap:13px;background:#222933;border:1px solid #35424e;border-radius:10px;padding:11px 14px}.data-flow-index{color:#55d6be;font-weight:700;font-size:.78rem}.data-flow-node strong{display:block;color:#f4f7fa}.data-flow-node span{display:block;color:#aab8c0;font-size:.78rem;margin-top:2px}.data-flow-arrow{text-align:center;font-size:1.2rem;line-height:1.35}
        .structure-diagram{background:#171b22;border:1px solid #303a45;border-radius:14px;padding:20px;text-align:center}.structure-root{display:inline-block;background:#315a63;border:1px solid #55d6be;border-radius:10px;padding:12px 18px;color:#f4f7fa;font-weight:700}.structure-branches{display:flex;justify-content:center;gap:18px;margin:18px 0}.structure-branch{width:230px;background:#222933;border:1px solid #35424e;border-radius:10px;padding:12px;color:#f4f7fa}.structure-branch b{display:block;color:#70f0d7}.structure-branch span{display:block;color:#aab8c0;font-size:.78rem;line-height:1.5;margin-top:5px}.structure-link{color:#e8a96a;font-size:.8rem;margin:5px 0 13px}.structure-record{display:flex;justify-content:center;gap:8px;flex-wrap:wrap}.structure-record span{background:#2b3340;color:#dce7eb;border-radius:6px;padding:7px 10px;font-size:.76rem}
        .imbalance-callout{background:#2b2325;border:1px solid #70454c;border-radius:12px;padding:16px;color:#d9c4c6;line-height:1.5}.imbalance-callout strong{color:#ff9b8e}
        @media(max-width:800px){.problem-illustration{flex-wrap:wrap;gap:12px}.illustration-arrow{transform:rotate(90deg)}.structure-branches{flex-direction:column;align-items:center}.structure-branch{width:min(100%,300px)}}
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_architecture_diagram():
    st.markdown(
        """
        <style>
        .architecture-diagram{background:linear-gradient(180deg,#171b22 0%,#13171d 100%);border:1px solid #303a45;border-radius:16px;padding:24px 20px 20px;margin:12px 0 24px}
        .architecture-stage{display:flex;flex-direction:column;align-items:center}
        .architecture-box{width:min(100%,620px);box-sizing:border-box;border:1px solid #40505d;border-radius:11px;background:#222933;padding:13px 18px;text-align:center;color:#f4f7fa;box-shadow:0 6px 16px #00000020}
        .architecture-box .box-kicker{display:block;color:#70f0d7;font-size:.68rem;letter-spacing:.13em;text-transform:uppercase;font-weight:700;margin-bottom:4px}
        .architecture-box .box-title{display:block;font-weight:700;font-size:1rem}
        .architecture-box .box-detail{display:block;color:#aab8c0;font-size:.76rem;margin-top:4px}
        .architecture-box.input{border-color:#55d6be;background:#1d343a}.architecture-box.backbone{border-color:#6f8de8;background:#252b45}.architecture-box.shared{border-color:#d29b5d;background:#382d24}
        .architecture-arrow{height:30px;display:flex;align-items:center;justify-content:center;color:#55d6be;font-size:1.35rem;line-height:1}
        .branch-label{font-size:.72rem;color:#aab8c0;text-align:center;margin:2px 0 10px;letter-spacing:.04em}
        .architecture-branches{width:min(100%,820px);display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;position:relative}
        .architecture-branches:before{content:"";position:absolute;top:-11px;left:25%;right:25%;height:1px;background:#66737d}
        .architecture-branch{position:relative;background:#222933;border:1px solid #40505d;border-radius:12px;padding:14px 15px 16px}
        .architecture-branch:before{content:"";position:absolute;top:-12px;left:50%;height:12px;border-left:1px solid #66737d}
        .architecture-branch.classification{border-top:3px solid #55d6be}.architecture-branch.localization{border-top:3px solid #ff806b}
        .architecture-branch h4{margin:0 0 4px;color:#f4f7fa;font-size:.95rem}.architecture-branch p{margin:0 0 12px;color:#aab8c0;font-size:.75rem}
        .branch-step{border-radius:8px;background:#2c3540;border:1px solid #3d4a55;padding:9px 10px;margin-top:8px;text-align:center;color:#e8f0f2;font-size:.78rem}
        .branch-step strong{display:block;color:#f4f7fa}.branch-step span{display:block;color:#aab8c0;font-size:.7rem;margin-top:3px}
        .branch-output{margin-top:12px;border-radius:8px;padding:11px;text-align:center;font-weight:700;font-size:.82rem}.classification .branch-output{background:#1d403d;color:#70f0d7}.localization .branch-output{background:#432d31;color:#ff9b8e}
        .architecture-legend{display:flex;justify-content:center;gap:18px;flex-wrap:wrap;margin-top:18px;color:#aab8c0;font-size:.72rem}.architecture-legend span:before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:#55d6be}.architecture-legend span:last-child:before{background:#ff806b}
        @media(max-width:650px){.architecture-branches{grid-template-columns:1fr}.architecture-branches:before{display:none}.architecture-branch:before{display:none}}
        </style>
        <div class="architecture-diagram">
          <div class="architecture-stage">
            <div class="architecture-box input"><span class="box-kicker">Input</span><span class="box-title">Chest X-ray</span><span class="box-detail">H × W × 3 &nbsp;|&nbsp; normalized grayscale copied across RGB channels</span></div>
            <div class="architecture-arrow">↓</div>
            <div class="architecture-box backbone"><span class="box-kicker">Feature extractor</span><span class="box-title">EfficientNetB0 / DenseNet121 backbone</span><span class="box-detail">ImageNet-pretrained convolutional features</span></div>
            <div class="architecture-arrow">↓</div>
            <div class="architecture-box shared"><span class="box-kicker">Shared representation</span><span class="box-title">GlobalAveragePooling → BatchNorm → Dense → Dropout</span><span class="box-detail">One compact feature vector feeds both task-specific heads</span></div>
            <div class="architecture-arrow">↓</div>
            <div class="branch-label">shared features split into two outputs</div>
            <div class="architecture-branches">
              <div class="architecture-branch classification">
                <h4>Classification head</h4><p>Answers: “Is pneumonia present?”</p>
                <div class="branch-step"><strong>Dense</strong><span>task-specific features</span></div>
                <div class="branch-step"><strong>Dropout</strong><span>regularization during training</span></div>
                <div class="branch-step"><strong>Dense(1, sigmoid)</strong><span>one probability between 0 and 1</span></div>
                <div class="branch-output">Pneumonia probability</div>
              </div>
              <div class="architecture-branch localization">
                <h4>Localization head</h4><p>Answers: “Where is it?”</p>
                <div class="branch-step"><strong>Dense</strong><span>task-specific features</span></div>
                <div class="branch-step"><strong>Dropout</strong><span>regularization during training</span></div>
                <div class="branch-step"><strong>Dense(4, sigmoid)</strong><span>four normalized values in [0, 1]</span></div>
                <div class="branch-output">[x, y, width, height]</div>
              </div>
            </div>
          </div>
          <div class="architecture-legend"><span>classification pathway</span><span>localization pathway</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- SIDEBAR ---
with st.sidebar:
    st.title(" PneumoScan AI")
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
# 01 · DATA
# ============================================================
with tab_data:
    render_data_styles()
    st.markdown("<div class='section-kicker'>01 / Data foundation</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='data-hero'><h2>From chest X-ray to clinical signal</h2>"
        "<p>PneumoScan AI is designed as a screening support system: it helps surface "
        "suspicious studies for human review, rather than replacing clinical judgment. "
        "The data story is a two-task problem — recognize <b>what</b> is present and "
        "show <b>where</b> it may be present.</p></div>",
        unsafe_allow_html=True,
    )

    st.subheader("What problem are we solving?")
    st.markdown(
        "Pneumonia is a lung infection that can appear on chest radiographs. Reviewing "
        "large volumes of X-rays is time-consuming, so an automated first pass can help "
        "medical professionals prioritize suspicious cases for closer examination."
    )
    render_problem_illustration()

    st.subheader("The RSNA dataset at a glance")
    st.markdown(
        "**RSNA** is the Radiological Society of North America. The RSNA Pneumonia "
        "Detection Challenge dataset is used here because it pairs chest X-ray studies "
        "with radiologist-derived pneumonia annotations. Each DICOM contains grayscale "
        "pixel data plus imaging metadata; the annotation CSV connects a patient/image "
        "identifier to a label and, for positive cases, box coordinates."
    )
    metric_cols = st.columns(4)
    metrics = [
        ("26,684", "total patients", "documented project figure"),
        ("~9,000", "pneumonia", "≈33% of patients"),
        ("~17,684", "normal", "≈67% of patients"),
        ("85 / 15", "train / validation", "patient-level, stratified"),
    ]
    for col, (value, label, note) in zip(metric_cols, metrics):
        col.markdown(
            f"<div class='metric-card'><div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>",
            unsafe_allow_html=True,
        )

    chart_col, explanation_col = st.columns([1.15, 1])
    with chart_col:
        distribution = go.Figure(
            go.Pie(
                labels=["Normal", "Pneumonia"],
                values=[67, 33],
                hole=0.68,
                marker=dict(colors=["#4b6470", "#ff806b"], line=dict(color="#171b22", width=4)),
                textinfo="label+percent",
                textfont=dict(color="#f4f7fa", size=13),
                hovertemplate="%{label}: %{percent}<extra></extra>",
            )
        )
        distribution.update_layout(
            title=dict(text="Patient-level class distribution", font=dict(size=16, color="#f4f7fa")),
            annotations=[dict(text="26,684<br><span style='font-size:11px'>patients</span>", x=0.5, y=0.5, showarrow=False, font=dict(color="#f4f7fa", size=18))],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=55, b=10, l=10, r=10),
            height=330,
            showlegend=False,
        )
        st.plotly_chart(distribution, use_container_width=True, config={"displayModeBar": False})
    with explanation_col:
        st.markdown("#### Why DICOM instead of JPG/PNG?")
        st.markdown(
            "DICOM is the clinical imaging standard. Unlike a screenshot, it preserves "
            "pixel data alongside acquisition metadata and a consistent study identity. "
            "The current inference path reads `pixel_array`, applies rescale slope/intercept "
            "when available, handles MONOCHROME1 inversion, then normalizes to 0–255."
        )

    st.subheader("How the dataset is organized")
    render_dataset_structure()

    st.subheader("From annotation to training target")
    render_data_flow([
        ("Original chest X-ray", "DICOM pixel array and patient/image identifier"),
        ("Annotation CSV", "Find the matching patient/image record"),
        ("Identify pneumonia annotations", "Positive rows carry one or more regions"),
        ("Extract box coordinates", "x, y, width, height from the annotation"),
        ("Create two targets", "binary class label + localization target"),
        ("Training-ready sample", "normalized image tensor and target dictionary"),
    ], accent="#ff806b")
    st.markdown(
        "A patient may have multiple boxes. The documented project convention merges "
        "them with the minimum x/y and maximum right/bottom extents. This produces one "
        "simple target box, but it loses the distinction between separate infected regions."
    )
    label_col, caveat_col = st.columns(2)
    with label_col:
        st.markdown("**Coordinate convention**")
        st.code("x_norm = x / image_width\ny_norm = y / image_height\nw_norm = width / image_width\nh_norm = height / image_height", language="python")

    st.subheader("Why class imbalance matters")
    imbalance_col, baseline_col = st.columns([1.1, 1])
    with imbalance_col:
        st.markdown(
            "Approximately two thirds of the documented patients are normal. A model "
            "that predicts **Normal for every study** could appear to achieve roughly "
            "**67% accuracy**, while detecting **0% of pneumonia cases**. For a screening "
            "workflow, that false sense of performance is unacceptable: recall, AUC, "
            "precision, and the false-negative trade-off matter more than accuracy alone."
        )
        st.markdown("<div class='imbalance-callout'><strong>Screening principle:</strong> a missed positive can delay care; the operating threshold should be chosen with that clinical cost in mind.</div>", unsafe_allow_html=True)
    with baseline_col:
        baseline = go.Figure(go.Bar(
            x=["All Normal baseline", "Pneumonia detected"],
            y=[67, 0],
            marker_color=["#4b6470", "#ff806b"],
            text=["67% accuracy", "0% recall"],
            textposition="auto",
        ))
        baseline.update_layout(
            title="Why accuracy alone misleads",
            yaxis=dict(range=[0, 75], title="Percent"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#dce7eb"), margin=dict(t=50, b=25, l=35, r=10), height=280,
        )
        st.plotly_chart(baseline, use_container_width=True, config={"displayModeBar": False})

    st.subheader("Patient-level train / validation split")
    render_data_flow([
        ("Complete dataset", "≈26,684 documented patients"),
        ("Patient-level stratified split", "Keep each patient in one partition; preserve class ratio"),
        ("Training: 85%", "Learn model parameters from labeled examples"),
        ("Validation: 15%", "Estimate generalization during development"),
    ])
    st.markdown(
        "Patient-level splitting prevents leakage: if multiple studies from one patient "
        "land in both partitions, the model may memorize patient-specific appearance "
        "instead of learning pneumonia patterns. Stratification keeps the approximate "
        "33% / 67% class balance comparable across training and validation. Exact sample "
        "counts are unavailable because the split artifact is not included."
    )

    st.subheader("Complete data pipeline")
    render_data_flow([
        ("RSNA dataset", "DICOM images + annotation records"),
        ("DICOM image loading", "Read pixel_array and relevant metadata"),
        ("Image preprocessing", "Rescale,CLAHE, invert MONOCHROME1, normalize, resize to 160×160, stack to 3 channels"),
        ("Annotation processing", "Map patient IDs and consolidate positive boxes"),
        ("Two training targets", "Pneumonia/Normal + normalized bounding box"),
        ("Patient-level split", "85% train / 15% validation, stratified"),
        ("Training-ready dataset", "Image tensor paired with both targets"),
        ("Deep learning model", "Shared backbone with classification and localization heads"),
    ], accent="#55d6be")
    st.success("Next: Architecture shows how the shared visual representation branches into the classification and localization heads.")

# ============================================================
# 03 · ARCHITECTURE
# ============================================================
with tab_arch:
    st.header("Model architecture")
    render_architecture_diagram()

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

    st.subheader("Two-phase transfer learning")
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

    st.info("Using model: BCE + validation-selected threshold (160px)")
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
            prob = st.session_state.last_prob
            bbox = st.session_state.last_bbox
            threshold = INFERENCE_THRESHOLD

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
            st.metric("Decision Threshold", f"{threshold:.0%}", "fixed validation-selected cutoff")
            st.metric("Confidence", confidence)

            if is_pneumonia and sum(bbox) > 0:
                st.info(f"Bounding Box (Normalized):\n`[{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]`\n\nThe bounding box highlights the localized area of suspected pneumonia.")

        with col1:
            st.subheader("Chest X-Ray")
            disp_img = draw_bbox_on_image(st.session_state.last_image, bbox, prob, threshold)
            st.image(disp_img, use_container_width=True)
