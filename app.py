import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ====================================================================
# 1. MINIMAL PLATFORM CONFIGURATION
# ====================================================================
st.set_page_config(
    page_title="DigitVision AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .title-text {
        text-align: center;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .subtitle-text {
        text-align: center;
        color: #888;
        margin-bottom: 30px;
    }
    .big-result {
        font-size: 5rem;
        font-weight: 800;
        text-align: center;
        letter-spacing: 8px;
        color: #38BDF8;
        line-height: 1.2;
    }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# 2. CACHED DEEP LEARNING INFERENCE RUNTIME
# ====================================================================
@st.cache_resource
def load_nn_model():
    return tf.keras.models.load_model("model.h5")

try:
    model = load_nn_model()
except Exception as e:
    st.error("Execution Alert: Could not find trained 'model.h5' parameters. Please compile train_model.py first.")

# ====================================================================
# 3. MAIN MINIMAL INTERFACE
# ====================================================================
st.markdown("<h1 class='title-text'>DigitVision AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Minimalist Multi-Digit Handwritten Recognition</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image containing handwritten digits", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.divider()
    pil_image = Image.open(uploaded_file)
    
    # --- COMPUTER VISION ADVANCED MULTI-SEGMENTATION LAYER (UNCHANGED) ---
    cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)
        
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_bounding_boxes = []
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        if w >= 4 and h >= 12:
            digit_bounding_boxes.append((x, y, w, h))
    
    digit_bounding_boxes = sorted(digit_bounding_boxes, key=lambda b: b[0])
    
    annotated_img = cv_img.copy()
    processed_digits = []
    
    for index, (x, y, w, h) in enumerate(digit_bounding_boxes):
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (56, 189, 248), 2)
        cv2.putText(annotated_img, f"#{index+1}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (129, 140, 248), 1)
        
        extracted_roi = thresh[y:y+h, x:x+w]
        
        pad_dim = max(w, h) + 14
        square_canvas = np.zeros((pad_dim, pad_dim), dtype=np.uint8)
        dx = (pad_dim - w) // 2
        dy = (pad_dim - h) // 2
        square_canvas[dy:dy+h, dx:dx+w] = extracted_roi
        
        digit_matrix = cv2.resize(square_canvas, (28, 28), interpolation=cv2.INTER_AREA)
        digit_matrix = digit_matrix / 255.0
        digit_matrix = digit_matrix.reshape(1, 28, 28, 1)
        
        prediction = model.predict(digit_matrix)
        predicted_lbl = np.argmax(prediction)
        confidence_val = np.max(prediction) * 100
        
        processed_digits.append({
            "label": predicted_lbl,
            "confidence": confidence_val,
            "probabilities": prediction[0] * 100,
            "crop": digit_matrix.reshape(28, 28)
        })
        
    # --- MINIMAL RESULTS DISPLAY ---
    if len(processed_digits) > 0:
        # Build Master Result String Array
        full_number_string = "".join([str(d["label"]) for d in processed_digits])
        
        st.markdown("<p style='text-align:center; color:#888; font-size: 14px; margin-bottom: 0;'>Resolved Sequence</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-result'>{full_number_string}</div>", unsafe_allow_html=True)
        
        st.image(annotated_img, caption="Computer Vision Segmentation", use_container_width=True)
        
        with st.expander("View Segment Details"):
            for i, d in enumerate(processed_digits):
                st.write(f"**Segment #{i+1}:** Predicted **{d['label']}** (Confidence: {d['confidence']:.1f}%)")
    else:
        st.warning("No clear digits were detected. Please try uploading a different image.")