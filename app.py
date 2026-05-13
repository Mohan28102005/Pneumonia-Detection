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

        # ✅ Use OpenCV resize exactly like notebook
        img_resized = cv2.resize(pixel_array, (image_size, image_size))

        # ✅ Stack to RGB exactly like notebook (no PIL conversion)
        img_rgb = np.stack([img_resized, img_resized, img_resized], axis=-1).astype(np.float32)

        # PIL image just for display (full resolution, no model input)
        pil_image = Image.fromarray(pixel_array)

        # Model input
        input_array = np.expand_dims(img_rgb, axis=0)  # (1, 160, 160, 3)

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
             bbox = [0.0, 0.0, 0.0, 0.0] # Dummy bbox
        else:
             st.warning(f"Unexpected output type. Returning 0.")
             prob = 0.0
             bbox = [0.0, 0.0, 0.0, 0.0]
             
        return prob, bbox
    except Exception as e:
        st.error(f"Inference error: {e}")
        return 0.0, [0.0, 0.0, 0.0, 0.0]

def draw_bbox_on_image(pil_image, bbox, probability, threshold):
    img_copy = pil_image.convert('RGB')  # ← convert here, not earlier
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
    **Research Dashboard & Live Inference**
    
    This app demonstrates a research journey in pneumonia detection from chest X-rays using dual-head models (classification + localization).
    """)
    st.markdown("---")

# --- TABS ---
tab1, tab2 = st.tabs(["Research Journey", "Live Inference"])

with tab1:
    st.header("Model Architecture Timeline")
    
    with st.expander("🧪 EXPERIMENT 1: EfficientNetB0 — FocalLoss alpha=0.25 (Warmup Phase)", expanded=True):
        st.error("Status: ❌ Failed")
        st.markdown("""
        **Architecture**:
        - Backbone: EfficientNetB0 (frozen during warmup)
        - Head 1: Classification (pneumonia yes/no)
        - Head 2: Localization (bounding box x,y,w,h)
        
        **Hyperparameters**:
        - Loss: FocalLoss(gamma=2, alpha=0.25) + HuberLoss
        - Optimizer: Adam
        - Image size: 224px
        - Oversampling: Yes (1:1 minority oversampling)
        
        **Mistake**: Alpha should have been ~0.5+ when oversampling. Kept at 0.25 by mistake.
        """)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Epoch 3 Val Loss", "0.1186", "-0.0799 vs Ep1")
        c2.metric("Val AUC", "0.5725")
        c3.metric("Precision", "0.0000")
        c4.metric("Recall", "0.0000")
        st.info("Observation: Recall stuck at 0. Model predicting all negatives. FocalLoss alpha too low for oversampled data — model penalizing positives insufficiently.")

    with st.expander("🧪 EXPERIMENT 2: EfficientNetB0 — FocalLoss alpha=0.75", expanded=False):
        st.warning("Status: ⚠️ Overcorrected")
        st.markdown("""
        **Architecture**: Same as Experiment 1  
        **Loss**: FocalLoss(gamma=2, alpha=0.75) + HuberLoss  
        **Change from prev**: alpha increased from 0.25 → 0.75
        """)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Epoch 2 Val Loss", "0.0779")
        c2.metric("Val AUC", "0.5721")
        c3.metric("Precision", "0.2473")
        c4.metric("Recall", "0.9974")
        st.info("Observation: Recall too high (0.9974 val) → model predicting almost everything as pneumonia. Alpha overcorrection. Not learning meaningful features.")

    with st.expander("🧪 EXPERIMENT 3: EfficientNetB0 — FocalLoss alpha=0.6 + Oversampling (Corrected)", expanded=False):
        st.warning("Status: ⚠️ Suboptimal")
        st.markdown("""
        **Architecture**: Same dual-head EfficientNetB0  
        **Loss**: FocalLoss(gamma=2, alpha=0.6) + HuberLoss  
        **Oversampling**: 1:1, each sample = (image, (classification_label, bounding_box))  
        **Change from prev**: alpha tuned to 0.6 — between 0.25 and 0.75
        """)
        st.info("Observation: Better balance attempted. Alpha=0.6 is the corrected sweet spot for oversampled data. Transition experiment toward BCE approach.")

    with st.expander("🧪 EXPERIMENT 4: EfficientNetB0 — BCE + HuberLoss + AdamW (160px)", expanded=False):
        st.warning("Status: ⚠️ Moderate")
        st.markdown("""
        **Architecture**: Same dual-head EfficientNetB0  
        **Loss**: BinaryCrossEntropy + HuberLoss  
        **Optimizer**: AdamW  
        **Image size**: 160px  
        **Val AUC**: 0.8335  
        
        **Prediction stats**: mean=0.358, std=0.189, min=0.032, max=0.967  
        *(22.7% above 0.5 | 55.3% above 0.3)*
        """)
        
        # Plot Threshold sweep
        thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        precision = [0.197, 0.227, 0.262, 0.289, 0.336, 0.373, 0.435, 0.464, 0.540, 0.588, 0.574]
        recall = [0.975, 0.955, 0.934, 0.881, 0.819, 0.708, 0.638, 0.506, 0.420, 0.317, 0.222]
        f1_scores = [0.328, 0.366, 0.409, 0.435, 0.476, 0.489, 0.518, 0.484, 0.472, 0.412, 0.320]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds, y=precision, mode='lines+markers', name='Precision'))
        fig.add_trace(go.Scatter(x=thresholds, y=recall, mode='lines+markers', name='Recall'))
        fig.add_trace(go.Scatter(x=thresholds, y=f1_scores, mode='lines+markers', name='F1 Score'))
        fig.update_layout(title="Threshold Sweep Metrics", xaxis_title="Threshold", yaxis_title="Score", hovermode="x unified", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("Issues: Low recall during training. Precision-recall imbalance at validation. At thresh=0.55: recall=0.63, precision=0.43 — catching pneumonia but too many FP.")
        st.code("Model path: /models/bce_threshold/best_model.keras")

    with st.expander("🧪 EXPERIMENT 5: DenseNet121 (512px) — RecallFocusedLoss + AdamW + 1:1 Oversampling", expanded=True):
        st.success("Status: ✅ Best")
        st.markdown("""
        **Architecture**:
        - Backbone: DenseNet121 (pretrained ImageNet)
        - Head 1: Classification (RecallFocusedLoss fn_weight=3.0)
        - Head 2: Localization (Huber delta=0.1)
        
        **Loss**: { classification: RecallFocusedLoss(fn_weight=3.0), localization: Huber(0.1) }  
        **Optimizer**: AdamW | **Oversampling**: 1:1 | **Image size**: 512px
        
        **Phase 1 — Warmup (10 epochs, LR=1e-4)**: Backbone frozen. Only classification + localization heads trained. Lets new layers learn task-specific features without disturbing pretrained weights.  
        **Phase 2 — Fine-tuning (LR=1e-5)**: Top 50 backbone layers unfrozen. Rest stays frozen. After reducing fn_weight: 3 → 2:
        """)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Epoch 7 Val Loss", "1.0212")
        c2.metric("Val AUC", "0.7962")
        c3.metric("Precision", "0.2194")
        c4.metric("Recall", "0.9300")
        st.code("Model path: /models/densenet_clinical/best_model.keras")

with tab2:
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
