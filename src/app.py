# app.py — Drowsiness detector (final decision), shows Mouth status; Eyes hidden by default.
# No new libraries. Class order fixed: ['Open','Closed','no_yawn','yawn'].

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import time
from collections import deque

# ---------- page ----------
st.set_page_config(page_title="Driver Drowsiness Detector", page_icon="😴", layout="centered")
st.title("😴 Driver Drowsiness Detection")
st.caption("Final decision = DROWSY / ALERT. Details show Mouth status. Eyes are hidden by default.")

# ---------- model ----------
@st.cache_resource
def load_drowsiness_model():
    return load_model("models/drowsiness_model.h5")

model = load_drowsiness_model()

# fixed class order you asked for
CLASSES = ['Open', 'Closed', 'no_yawn', 'yawn']

# ---------- sidebar ----------
st.sidebar.header("⚙️ Decision Settings")
use_eyes_in_decision = st.sidebar.checkbox("Also use Eyes in decision (hidden in UI)", value=False)

# thresholds (tune if needed)
EYE_THRESHOLD   = st.sidebar.slider("Eyes-closed threshold (Closed vs Open)", 0.00, 1.00, 0.60, 0.01)
YAWN_THRESHOLD  = st.sidebar.slider("Mouth-open threshold (yawn vs no_yawn)", 0.00, 1.00, 0.75, 0.01)

# bias + uncertainty to reduce false 'yawn'
yawn_bias_factor = st.sidebar.slider("Reduce yawn bias (multiply yawn prob by …)", 0.50, 1.00, 0.85, 0.01)
uncertainty_gap  = st.sidebar.slider("Uncertainty margin (treat close calls as closed)", 0.00, 0.20, 0.05, 0.01)

use_face_crop = st.sidebar.checkbox("Crop to face before predicting (recommended)", value=True)
mode = st.sidebar.radio("Input mode", ["Upload Image", "Use Webcam"], index=0)
st.sidebar.markdown("---")
st.sidebar.caption("If mouth still looks wrong, raise Mouth threshold to 0.80–0.85, and/or reduce yawn bias to ~0.75.")

# ---------- helpers ----------
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def crop_face(pil_img, expand=0.20):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
    if len(faces) == 0:
        return pil_img
    x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
    H,W = img.shape[:2]
    x0 = max(0, int(x - expand*w)); y0 = max(0, int(y - expand*h))
    x1 = min(W, int(x + w*(1+expand))); y1 = min(H, int(y + h*(1+expand)))
    return Image.fromarray(img[y0:y1, x0:x1])

def preprocess(pil_img, size=(224,224)):
    if use_face_crop:
        pil_img = crop_face(pil_img)
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

def predict_probs(pil_img):
    """Return normalized class probs in our fixed order and display image."""
    x, disp = preprocess(pil_img)
    raw = model.predict(x, verbose=0)[0]  # expect 4 scores
    probs = to_probs(raw)
    if len(probs) < 4:
        st.error("Model must output 4 scores: ['Open','Closed','no_yawn','yawn'].")
        st.stop()
    # map by our fixed order
    p_open   = float(probs[0])
    p_closed = float(probs[1])
    p_noyawn = float(probs[2])
    p_yawn   = float(probs[3])
    return {'Open': p_open, 'Closed': p_closed, 'no_yawn': p_noyawn, 'yawn': p_yawn}, disp

def eye_closed_prob(p_open, p_closed):
    return p_closed / (p_closed + p_open + 1e-7)

def mouth_open_prob(p_noyawn, p_yawn, bias=1.0):
    # apply bias to reduce over-confident yawn
    p_yawn_adj = max(0.0, min(1.0, p_yawn * bias))
    return p_yawn_adj / (p_yawn_adj + p_noyawn + 1e-7)

def decide_from_probs(probs):
    # compute pairwise probabilities
    eye_p   = eye_closed_prob(probs['Open'], probs['Closed'])
    mouth_p = mouth_open_prob(probs['no_yawn'], probs['yawn'], bias=yawn_bias_factor)

    # apply uncertainty margin around threshold → treat as safe/closed
    mouth_open = mouth_p > (YAWN_THRESHOLD + uncertainty_gap)
    eye_closed = False
    if use_eyes_in_decision:
        eye_closed = eye_p > (EYE_THRESHOLD + uncertainty_gap)

    is_drowsy = mouth_open or eye_closed
    return is_drowsy, mouth_open, eye_closed, mouth_p, eye_p

def show_mouth_details(mouth_p):
    st.subheader("🔎 Inference Details (Mouth)")
    st.write(f"**Mouth Open probability (yawn vs no_yawn):** {mouth_p:.2f}  — threshold: {YAWN_THRESHOLD:.2f}")
    st.progress(float(mouth_p))

def show_class_bars(probs):
    st.write("**Raw class probabilities (normalized):**")
    for k in ['Open','Closed','no_yawn','yawn']:
        v = float(probs[k])
        st.write(f"- {k}: {v:.2f}")
        st.progress(float(min(max(v,0.0),1.0)))

# ---------- upload mode ----------
if mode == "Upload Image":
    file = st.file_uploader("Upload a driver's face image:", type=["jpg","jpeg","png"])
    if file:
        pil = Image.open(file).convert("RGB")
        probs, disp_img = predict_probs(pil)

        st.image(disp_img, caption="Used for prediction", use_container_width=True)

        is_drowsy, mouth_open, eye_closed, mouth_p, eye_p = decide_from_probs(probs)

        st.markdown("---")
        # final decision
        if is_drowsy:
            st.error("⚠️ DROWSY")
        else:
            st.success("✅ ALERT")

        # details (mouth only, as requested)
        show_mouth_details(mouth_p)
        st.markdown("— Mouth status: **{}**".format("Open (Yawn)" if mouth_open else "Closed (no_yawn)"))

        # diagnostics (optional; eyes not shown but still available if you toggled them in decision)
        with st.expander("Diagnostics (raw probabilities)"):
            show_class_bars(probs)
            st.write(f"(Hidden) Eyes-closed probability: {eye_p:.2f}; threshold: {EYE_THRESHOLD:.2f}")

# ---------- webcam mode ----------
else:
    st.subheader("🎥 Real-time Detection")
    st.info("Grant permission, then tick 'Start'. Good lighting helps.")
    start = st.checkbox("Start")
    FRAME = st.image([], caption="Webcam Feed", use_container_width=True)

    # smooth mouth probability to avoid flicker
    window = deque(maxlen=7)  # ~0.2s smoothing at 30 FPS

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

        probs, _ = predict_probs(pil)
        is_drowsy, mouth_open, eye_closed, mouth_p, eye_p = decide_from_probs(probs)

        window.append(mouth_p)
        mouth_p_smooth = float(np.mean(window))

        label = "DROWSY" if is_drowsy else "ALERT"
        color = (255, 0, 0) if is_drowsy else (0, 200, 0)
        text = f"{label} | MouthOpen {mouth_p_smooth:.2f}"
        cv2.putText(rgb, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        FRAME.image(rgb)
        time.sleep(0.03)

    if cap is not None:
        cap.release()
