# app.py — Drowsiness by mouth (yawn vs no_yawn), improved with mouth ROI + averaging
# No new packages. Class order fixed: ['Open','Closed','no_yawn','yawn'].

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance
import cv2
import time
from collections import deque

# -------------------- PAGE --------------------
st.set_page_config(page_title="Driver Drowsiness Detector", page_icon="😴", layout="centered")
st.title("😴 Driver Drowsiness Detection")
st.caption("Decision uses **mouth (yawn vs no_yawn)**. Eyes are hidden by default. Class order fixed: ['Open','Closed','no_yawn','yawn'].")

# -------------------- MODEL --------------------
@st.cache_resource
def load_drowsiness_model():
    return load_model("models/drowsiness_model.h5")

model = load_drowsiness_model()

# fixed order
CLASSES = ['Open', 'Closed', 'no_yawn', 'yawn']

# -------------------- SIDEBAR --------------------
st.sidebar.header("⚙️ Decision Settings")

YAWN_THRESHOLD  = st.sidebar.slider("Mouth-open threshold (yawn vs no_yawn)", 0.00, 1.00, 0.75, 0.01)
yawn_bias       = st.sidebar.slider("Reduce yawn bias (multiply yawn prob by…)", 0.50, 1.00, 0.85, 0.01)
uncertainty_gap = st.sidebar.slider("Uncertainty margin (treat close calls as closed)", 0.00, 0.20, 0.05, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Mouth ROI (relative to detected face)")
roi_top     = st.sidebar.slider("Top of mouth region (0=top face, 1=bottom)", 0.30, 0.80, 0.58, 0.01)
roi_height  = st.sidebar.slider("Height of mouth region (fraction of face)",   0.15, 0.60, 0.32, 0.01)
roi_margin  = st.sidebar.slider("Extra side margin (fraction of face width)",  0.00, 0.30, 0.08, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Stabilization")
tta_flips   = st.sidebar.checkbox("Average with horizontal flip", value=True)
tta_bright  = st.sidebar.checkbox("Average with slight brightness ±10%", value=True)

use_face_crop = st.sidebar.checkbox("Fallback: if no face, use entire image", value=True)
mode = st.sidebar.radio("Input mode", ["Upload Image", "Use Webcam"], index=0)
st.sidebar.caption("Tip: If it still flags closed mouths, raise threshold to 0.80–0.85 and lower bias to ~0.75.")

# -------------------- HELPERS --------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face_bbox(rgb):
    """Returns (x,y,w,h) for the largest face; None if not found."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2]*f[3])

def crop_mouth_roi(pil_img):
    """Crop lower face (mouth ROI) from PIL image using sliders. Fallbacks to face or whole image."""
    rgb = np.array(pil_img.convert("RGB"))
    H, W = rgb.shape[:2]
    bbox = detect_face_bbox(rgb)

    if bbox is None:
        if not use_face_crop:
            return pil_img
        # fallback: whole image as 'face'
        x, y, w, h = 0, 0, W, H
    else:
        x, y, w, h = bbox

    # compute mouth ROI within the face bbox
    top    = y + int(roi_top * h)
    height = int(roi_height * h)
    bottom = min(H, top + height)
    left   = max(0, x - int(roi_margin * w))
    right  = min(W, x + w + int(roi_margin * w))

    mouth = rgb[top:bottom, left:right]
    # safety fallback
    if mouth.size == 0:
        mouth = rgb[max(0, y + int(0.55*h)): min(H, y + int(0.9*h)), x: x + w]
        if mouth.size == 0:
            return pil_img
    return Image.fromarray(mouth)

def preprocess(pil_img, size=(224,224)):
    im = pil_img.resize(size).convert("RGB")
    arr = np.array(im).astype("float32") / 255.0
    x = np.expand_dims(arr, axis=0)
    return x, im

def to_probs(vec):
    v = np.array(vec, dtype="float64")
    s = v.sum()
    if 0.98 <= s <= 1.02:
        return v
    v = np.exp(v - v.max())  # softmax safety
    return v / v.sum()

def mouth_open_prob_from_pil(pil_img):
    """Compute mouth-open prob using mouth ROI + optional averaging."""
    roi = crop_mouth_roi(pil_img)

    variants = [roi]
    if tta_flips:
        variants.append(roi.transpose(Image.FLIP_LEFT_RIGHT))
    if tta_bright:
        variants.append(ImageEnhance.Brightness(roi).enhance(0.9))
        variants.append(ImageEnhance.Brightness(roi).enhance(1.1))

    probs = []
    for im in variants:
        x, disp = preprocess(im)
        raw = model.predict(x, verbose=0)[0]
        p = to_probs(raw)
        p_noyawn = float(p[2])
        p_yawn   = float(p[3]) * yawn_bias
        mouth_p  = p_yawn / (p_yawn + p_noyawn + 1e-7)
        probs.append(mouth_p)

    return float(np.mean(probs)), roi  # average score, and the ROI we used

def decide(mouth_p):
    mouth_open = mouth_p > (YAWN_THRESHOLD + uncertainty_gap)
    return mouth_open, ("Open (Yawn)" if mouth_open else "Closed (no_yawn)")

def show_mouth_details(mouth_p):
    st.subheader("🔎 Inference Details (Mouth)")
    st.write(f"**Mouth Open probability (yawn vs no_yawn):** {mouth_p:.2f}  — threshold: {YAWN_THRESHOLD:.2f}")
    st.progress(float(mouth_p))

# -------------------- UPLOAD MODE --------------------
if mode == "Upload Image":
    file = st.file_uploader("Upload a driver's face image:", type=["jpg","jpeg","png"])
    if file:
        pil = Image.open(file).convert("RGB")

        mouth_p, roi_img = mouth_open_prob_from_pil(pil)

        # show ROI actually used
        st.image(roi_img, caption="Mouth ROI used for prediction", use_container_width=True)

        mouth_open, mouth_text = decide(mouth_p)
        is_drowsy = mouth_open

        st.markdown("---")
        st.subheader("Final Decision")
        if is_drowsy:
            st.error("⚠️ DROWSY")
        else:
            st.success("✅ ALERT")

        show_mouth_details(mouth_p)
        st.markdown(f"— Mouth status: **{mouth_text}**")

# -------------------- WEBCAM MODE --------------------
else:
    st.subheader("🎥 Real-time Detection")
    st.info("Grant permission, then tick 'Start'. Good lighting helps.")
    start = st.checkbox("Start")
    FRAME = st.image([], caption="Webcam Feed", use_container_width=True)

    smooth = deque(maxlen=7)  # temporal smoothing

    cap = None
    if start:
        cap = cv2.VideoCapture(0)

    while start:
        ok, frame = cap.read()
        if not ok:
            st.warning("Unable to read from webcam.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        mouth_p, _ = mouth_open_prob_from_pil(pil)
        smooth.append(mouth_p)
        mouth_p_s = float(np.mean(smooth))

        mouth_open, _ = decide(mouth_p_s)
        label = "DROWSY" if mouth_open else "ALERT"
        color = (255, 0, 0) if mouth_open else (0, 200, 0)
        text = f"{label} | MouthOpen {mouth_p_s:.2f}"
        cv2.putText(rgb, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        FRAME.image(rgb)
        time.sleep(0.03)

    if cap is not None:
        cap.release()
