# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import time

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Driver Drowsiness Detector 🚗", page_icon="😴", layout="centered")
st.title("🚗 Driver Drowsiness & Yawn Detection")
st.caption("Class order is fixed to: ['Open','Closed','no_yawn','yawn']")

# ===================== MODEL =====================
@st.cache_resource
def load_drowsiness_model():
    # expects: models/drowsiness_model.h5
    return load_model("models/drowsiness_model.h5")

model = load_drowsiness_model()

# FIXED class order as requested (do not change)
CLASSES = ['Open', 'Closed', 'no_yawn', 'yawn']

# ===================== SIDEBAR =====================
st.sidebar.header("⚙️ Settings")
EYE_THRESHOLD   = st.sidebar.slider("Eyes closed threshold (Closed vs Open)", 0.00, 1.00, 0.55, 0.01)
YAWN_THRESHOLD  = st.sidebar.slider("Mouth open threshold (yawn vs no_yawn)", 0.00, 1.00, 0.55, 0.01)
use_face_crop   = st.sidebar.checkbox("Crop to face before predicting (recommended)", value=True)

mode = st.sidebar.radio("Input mode", ["Upload Image", "Use Webcam"], index=0)
st.sidebar.markdown("---")
st.sidebar.caption("If it still seems over-sensitive, raise the thresholds a bit (e.g., 0.65).")

# ===================== HELPERS =====================
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def crop_face(pil_img, expand=0.20):
    """Detect largest face and return padded crop; fallback to original if none found."""
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return pil_img
    # largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    H, W = img.shape[:2]
    x0 = max(0, int(x - expand * w)); y0 = max(0, int(y - expand * h))
    x1 = min(W, int(x + w * (1 + expand))); y1 = min(H, int(y + h * (1 + expand)))
    face = img[y0:y1, x0:x1]
    return Image.fromarray(face)

def preprocess_pil(pil_img, size=(224, 224)):
    """(Optional) face crop -> resize -> scale to [0,1] -> (1,H,W,3)."""
    if use_face_crop:
        pil_img = crop_face(pil_img)
    im = pil_img.resize(size).convert("RGB")
    arr = np.array(im).astype("float32") / 255.0
    x = np.expand_dims(arr, axis=0)
    return x, im  # model tensor + display image

def to_probs(vec):
    """Ensure we have probabilities: if sum ~1, use as-is; else softmax."""
    v = np.array(vec, dtype="float64")
    s = v.sum()
    if 0.98 <= s <= 1.02:
        return v
    # Softmax safety for logits/weird outputs
    v = np.exp(v - v.max())
    v = v / v.sum()
    return v

def predict_from_pil(pil_img):
    """
    Returns:
      eye_closed_prob (Closed vs Open),
      mouth_open_prob (yawn vs no_yawn),
      class_probs_dict (normalized),
      image_used_for_pred (PIL)
    """
    x, disp_img = preprocess_pil(pil_img)
    raw = model.predict(x, verbose=0)[0]  # shape should be (4,)
    probs = to_probs(raw)

    # map to our fixed class order
    # indices: 0:Open, 1:Closed, 2:no_yawn, 3:yawn
    if len(probs) < 4:
        st.error("Model does not output 4 scores. Check your .h5 file.")
        st.stop()

    p_open    = float(probs[0])
    p_closed  = float(probs[1])
    p_noyawn  = float(probs[2])
    p_yawn    = float(probs[3])

    # pairwise relative probabilities (stable)
    eye_closed_prob  = p_closed / (p_closed + p_open + 1e-7)
    mouth_open_prob  = p_yawn   / (p_yawn   + p_noyawn + 1e-7)

    cls_map = {
        'Open': p_open, 'Closed': p_closed,
        'no_yawn': p_noyawn, 'yawn': p_yawn
    }
    return eye_closed_prob, mouth_open_prob, cls_map, disp_img

def show_prob_bars(title, mapping):
    st.markdown(f"**{title}**")
    for k in ['Open','Closed','no_yawn','yawn']:  # keep consistent order
        v = float(mapping.get(k, 0.0))
        st.write(f"- {k}: {v:.2f}")
        st.progress(float(min(max(v, 0.0), 1.0)))

# ===================== UPLOAD IMAGE =====================
if mode == "Upload Image":
    file = st.file_uploader("Upload a driver's face image:", type=["jpg","jpeg","png"])
    if file:
        pil = Image.open(file).convert("RGB")
        eye_p, mouth_p, cls_map, disp_img = predict_from_pil(pil)

        st.image(disp_img, caption="Used for prediction", use_container_width=True)

        # decisions
        eye_closed = eye_p   > EYE_THRESHOLD
        mouth_open = mouth_p > YAWN_THRESHOLD
        is_drowsy  = eye_closed or mouth_open

        st.markdown("---")
        st.subheader("🔎 Inference Details")

        show_prob_bars("Raw class probabilities (normalized)", cls_map)

        st.write(f"**Eyes Closed (Closed vs Open):** {eye_p:.2f}  —  threshold: {EYE_THRESHOLD:.2f}")
        st.progress(float(eye_p))
        st.write(f"**Mouth Open (yawn vs no_yawn):** {mouth_p:.2f}  —  threshold: {YAWN_THRESHOLD:.2f}")
        st.progress(float(mouth_p))

        st.markdown("---")
        if is_drowsy:
            st.error("⚠️ DROWSY — triggered by **Eyes Closed** or **Mouth Open (Yawn)**")
        else:
            st.success("✅ ALERT — within safe thresholds")

# ===================== WEBCAM =====================
else:
    st.subheader("🎥 Real-time Detection via Webcam")
    st.info("Grant permission, then tick 'Start'. For best results, face the camera with good lighting.")
    start = st.checkbox("Start")
    FRAME = st.image([], caption="Webcam Feed", use_container_width=True)

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

        eye_p, mouth_p, cls_map, _ = predict_from_pil(pil)
        eye_closed = eye_p   > EYE_THRESHOLD
        mouth_open = mouth_p > YAWN_THRESHOLD
        is_drowsy  = eye_closed or mouth_open

        label = "DROWSY" if is_drowsy else "ALERT"
        color = (255, 0, 0) if is_drowsy else (0, 200, 0)
        text = f"{label} | EyesClosed {eye_p:.2f} | MouthOpen {mouth_p:.2f}"
        cv2.putText(rgb, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        FRAME.image(rgb)
        time.sleep(0.03)

    if cap is not None:
        cap.release()
