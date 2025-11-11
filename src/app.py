# app.py — Drowsiness by mouth only, no settings UI. Conservative + ROI + darkness heuristic.
# Fixed class order: ['Open','Closed','no_yawn','yawn'].
# No new dependencies beyond TF/Keras, PIL, NumPy, OpenCV.

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance
import cv2
import time
from collections import deque

# -------------------- CONSTANTS (tuned conservative) --------------------
# Final label = DROWSY iff we believe mouth is really open.
YAWN_THRESHOLD   = 0.90   # model’s mouth-open (yawn) prob must exceed this (after bias)
YAWN_BIAS        = 0.70   # multiply model yawn prob by this to damp false positives
DARK_MIN_RATIO   = 0.18   # central mouth band must have at least this fraction of dark pixels
DARK_PIX_THRESH  = 80     # darkness threshold for pixels (0..255)
UNCERTAINTY_GAP  = 0.08   # margin; if close to threshold we treat as closed

# Mouth ROI (as fraction of detected face box)
ROI_TOP_FRACTION    = 0.60  # start of ROI measured from top of face (0..1)
ROI_HEIGHT_FRACTION = 0.24  # vertical size of ROI as fraction of face height
ROI_SIDE_MARGIN     = 0.06  # extra margin on both sides as fraction of face width

# Test-time averaging for stability
USE_FLIP_AVG    = True
USE_BRIGHT_AVG  = True  # brightness +/- 10%

# -------------------- PAGE --------------------
st.set_page_config(page_title="Driver Drowsiness Detector", page_icon="😴", layout="centered")
st.title("😴 Driver Drowsiness Detection (Mouth Only)")
st.caption("Conservative detector with mouth ROI + darkness heuristic. Eyes are not used.")

# -------------------- MODEL --------------------
@st.cache_resource
def load_drowsiness_model():
    return load_model("models/drowsiness_model.h5")

model = load_drowsiness_model()
CLASSES = ['Open', 'Closed', 'no_yawn', 'yawn']

# -------------------- FACE/MOUTH HELPERS --------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face_bbox(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2]*f[3])

def crop_mouth_roi(pil_img):
    """Crop lower face mouth ROI using constants; fallback to whole image if no face."""
    rgb = np.array(pil_img.convert("RGB"))
    H, W = rgb.shape[:2]
    bbox = detect_face_bbox(rgb)
    if bbox is None:
        # fallback: bottom quarter of the image centered
        top = int(0.60 * H)
        bottom = min(H, int(top + 0.25 * H))
        left = int(0.15 * W)
        right = int(0.85 * W)
        mouth = rgb[top:bottom, left:right]
        if mouth.size == 0:
            return pil_img
        return Image.fromarray(mouth)

    x, y, w, h = bbox
    top    = y + int(ROI_TOP_FRACTION * h)
    height = int(ROI_HEIGHT_FRACTION * h)
    bottom = min(H, top + height)
    left   = max(0, x - int(ROI_SIDE_MARGIN * w))
    right  = min(W, x + w + int(ROI_SIDE_MARGIN * w))
    mouth  = rgb[top:bottom, left:right]
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
    v = np.exp(v - v.max())
    return v / v.sum()

def central_dark_ratio(pil_img):
    """Return fraction of dark pixels in central band of the mouth ROI."""
    img = np.array(pil_img.convert("RGB"))
    H, W = img.shape[:2]
    # central horizontal band (middle 40% height)
    y0 = int(0.30 * H); y1 = int(0.70 * H)
    x0 = int(0.15 * W); x1 = int(0.85 * W)
    band = img[y0:y1, x0:x1]
    if band.size == 0:
        return 0.0
    gray = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY)
    dark = (gray < DARK_PIX_THRESH).astype(np.uint8)
    return float(dark.mean())  # 0..1

def mouth_open_prob_from_pil(pil_img):
    roi = crop_mouth_roi(pil_img)

    variants = [roi]
    if USE_FLIP_AVG:
        variants.append(roi.transpose(Image.FLIP_LEFT_RIGHT))
    if USE_BRIGHT_AVG:
        variants.append(ImageEnhance.Brightness(roi).enhance(0.9))
        variants.append(ImageEnhance.Brightness(roi).enhance(1.1))

    ps = []
    for im in variants:
        x, _ = preprocess(im)
        raw = model.predict(x, verbose=0)[0]     # 4 scores
        p = to_probs(raw)
        p_noyawn = float(p[2])
        p_yawn   = float(p[3]) * YAWN_BIAS
        ps.append(p_yawn / (p_yawn + p_noyawn + 1e-7))

    p_mouth = float(np.mean(ps))
    return p_mouth, roi

def decide_mouth_open(p_mouth, dark_ratio):
    # Conservative: must pass threshold AND darkness
    open_by_model = p_mouth > (YAWN_THRESHOLD + UNCERTAINTY_GAP)
    open_by_dark  = dark_ratio >= DARK_MIN_RATIO
    return bool(open_by_model and open_by_dark)

# -------------------- UI LOGIC --------------------
mode = st.radio("Choose Input", ["Upload Image", "Use Webcam"], index=0, horizontal=True)

if mode == "Upload Image":
    file = st.file_uploader("Upload a driver's face image:", type=["jpg","jpeg","png"])
    if file:
        pil = Image.open(file).convert("RGB")

        p_mouth, roi_img = mouth_open_prob_from_pil(pil)
        dr = central_dark_ratio(roi_img)
        mouth_open = decide_mouth_open(p_mouth, dr)
        is_drowsy = mouth_open

        st.image(roi_img, caption="Mouth ROI used for prediction", use_container_width=True)

        st.markdown("### Final Decision")
        if is_drowsy:
            st.error("⚠️ DROWSY")
        else:
            st.success("✅ ALERT")

        st.markdown("### Inference Details (Mouth)")
        st.write(f"**Model mouth-open probability (avg):** {p_mouth:.2f}  — threshold: {YAWN_THRESHOLD:.2f}")
        st.progress(float(p_mouth))
        st.write(f"**Dark pixel ratio (central band):** {dr:.2f}  — required ≥ {DARK_MIN_RATIO:.2f}")
        st.write(f"— Mouth status: **{'Open (Yawn)' if mouth_open else 'Closed (no_yawn)'}**")

else:
    st.info("Grant permission, then tick **Start**. Good lighting helps.")
    start = st.checkbox("Start")
    FRAME = st.image([], caption="Webcam Feed", use_container_width=True)
    smooth = deque(maxlen=7)

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

        p_mouth, _ = mouth_open_prob_from_pil(pil)
        dr = central_dark_ratio(crop_mouth_roi(pil))
        smooth.append(p_mouth)
        p_avg = float(np.mean(smooth))

        mouth_open = decide_mouth_open(p_avg, dr)
        label = "DROWSY" if mouth_open else "ALERT"
        color = (255, 0, 0) if mouth_open else (0, 200, 0)
        text = f"{label} | p_mouth {p_avg:.2f} | dark {dr:.2f}"
        cv2.putText(rgb, text, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        FRAME.image(rgb)
        time.sleep(0.03)

    if cap is not None:
        cap.release()
