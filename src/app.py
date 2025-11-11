import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

st.set_page_config(page_title="Driver Drowsiness Detector 🚗", page_icon="😴", layout="centered")
st.title("🚗 Driver Drowsiness & Yawn Detection")
st.caption("Adaptive inference: handles 1, 2, or 4 output models. Tune thresholds on the left.")

# ---------------- Load model ----------------
@st.cache_resource
def load_drowsiness_model():
    m = load_model("models/drowsiness_model.h5")
    return m

model = load_drowsiness_model()

# Infer output size (1, 2, or 4)
try:
    out_dim = int(model.output_shape[-1])
except Exception:
    out_dim = 1  # safe default

# ---------------- Sidebar controls ----------------
st.sidebar.header("⚙️ Controls")

# Class mapping options for 2- or 4-class models
two_class_maps = {
    "['Open','Closed']": ['Open','Closed'],
    "['Closed','Open']": ['Closed','Open'],
    "['alert','drowsy']": ['alert','drowsy'],
    "['drowsy','alert']": ['drowsy','alert'],
}
four_class_maps = {
    "['Closed','Open','no_yawn','yawn']": ['Closed','Open','no_yawn','yawn'],
    "['Open','Closed','no_yawn','yawn']": ['Open','Closed','no_yawn','yawn'],
    "['Closed','Open','yawn','no_yawn']": ['Closed','Open','yawn','no_yawn'],
    "['Open','Closed','yawn','no_yawn']": ['Open','Closed','yawn','no_yawn'],
}

if out_dim == 1:
    st.sidebar.markdown("**Detected model type:** 1 output (binary sigmoid)")
    drowsy_threshold = st.sidebar.slider("Drowsy probability threshold", 0.0, 1.0, 0.60, 0.01)
    label_when_high = st.sidebar.selectbox("What does HIGH mean?", ["drowsy", "closed"], index=0)
elif out_dim == 2:
    st.sidebar.markdown("**Detected model type:** 2 outputs (softmax)")
    class_order = st.sidebar.selectbox("Pick class order", list(two_class_maps.keys()), index=0)
    CLASSES = two_class_maps[class_order]
    drowsy_label = st.sidebar.selectbox("Which label means Drowsy?", [CLASSES[0], CLASSES[1]], index=1)
    drowsy_threshold = st.sidebar.slider("Drowsy threshold", 0.0, 1.0, 0.60, 0.01)
else:
    st.sidebar.markdown("**Detected model type:** 4 outputs (softmax)")
    class_order = st.sidebar.selectbox("Pick class order", list(four_class_maps.keys()), index=0)
    CLASSES = four_class_maps[class_order]
    drowsy_components = st.sidebar.multiselect(
        "Treat these as Drowsy", ["Closed","yawn","drowsy"], default=["Closed","yawn"]
    )
    eye_thresh = st.sidebar.slider("Closed threshold", 0.0, 1.0, 0.55, 0.01)
    yawn_thresh = st.sidebar.slider("Yawn threshold", 0.0, 1.0, 0.60, 0.01)

use_face_crop = st.sidebar.checkbox("Crop to face", value=True)
prep_mode = st.sidebar.selectbox("Preprocessing", ["0-1 rescale", "ImageNet mean/std"], index=0)

# --------------- Helpers ---------------
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def crop_face(pil_img, expand=0.2):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
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
    pil_img = pil_img.resize(size)
    arr = np.array(pil_img).astype("float32")
    if prep_mode == "0-1 rescale":
        arr = arr/255.0
    else:
        arr = arr/255.0
        mean = np.array([0.485,0.456,0.406], dtype="float32")
        std  = np.array([0.229,0.224,0.225], dtype="float32")
        arr = (arr - mean)/std
    x = np.expand_dims(arr, axis=0)
    return x, pil_img

def softmax_safe(v):
    v = v.astype("float64")
    v = np.exp(v - v.max())
    return v / v.sum()

def sigmoid(v):
    return 1.0/(1.0+np.exp(-v))

def predict_probs(x):
    raw = model.predict(x, verbose=0)[0]
    # handle shapes safely
    if np.ndim(raw) == 0:
        raw = np.array([raw], dtype="float64")
    if out_dim == 1:
        p = float(raw[0])
        # if looks like a logit, pass through sigmoid; if already prob, clamp
        p = sigmoid(p) if not (0.0 <= p <= 1.0) else p
        return {"drowsy_prob": p}
    elif out_dim == 2:
        probs = raw if 0.98 <= raw.sum() <= 1.02 else softmax_safe(raw)
        return {CLASSES[i]: float(probs[i]) for i in range(2)}
    else:
        probs = raw if 0.98 <= raw.sum() <= 1.02 else softmax_safe(raw)
        return {CLASSES[i]: float(probs[i]) for i in range(min(4, len(probs)))}

def decide(probs):
    """Return (is_drowsy, detail_string)."""
    if out_dim == 1:
        p = probs["drowsy_prob"]
        is_d = (p >= drowsy_threshold)
        return is_d, f"drowsy_prob={p:.2f} (≥ {drowsy_threshold:.2f} → Drowsy)"
    elif out_dim == 2:
        p = probs.get(drowsy_label, 0.0)
        is_d = (p >= drowsy_threshold)
        return is_d, f"{drowsy_label}={p:.2f} (≥ {drowsy_threshold:.2f} → Drowsy)"
    else:
        p_closed = probs.get("Closed", 0.0)
        p_yawn   = probs.get("yawn",   0.0)
        parts = []
        is_d = False
        if "Closed" in drowsy_components:
            parts.append(f"Closed={p_closed:.2f} (≥ {eye_thresh:.2f})")
            is_d |= (p_closed >= eye_thresh)
        if "yawn" in drowsy_components:
            parts.append(f"yawn={p_yawn:.2f} (≥ {yawn_thresh:.2f})")
            is_d |= (p_yawn >= yawn_thresh)
        if "drowsy" in drowsy_components:
            p_d = probs.get("drowsy", 0.0)
            parts.append(f"drowsy={p_d:.2f} (≥ {max(eye_thresh,yawn_thresh):.2f})")
            is_d |= (p_d >= max(eye_thresh, yawn_thresh))
        return is_d, " | ".join(parts)

# ---------------- UI ----------------
mode = st.radio("Input mode", ["Upload Image", "Use Webcam"], horizontal=True)

if mode == "Upload Image":
    file = st.file_uploader("Upload a driver's face image:", type=["jpg","jpeg","png"])
    if file:
        pil = Image.open(file).convert("RGB")
        x, vis = preprocess(pil)
        st.image(vis, caption="Used for prediction", use_container_width=True)

        probs = predict_probs(x)
        is_drowsy, detail = decide(probs)

        # show raw outputs
        st.markdown("### 🔎 Probabilities")
        for k, v in probs.items():
            st.write(f"**{k}**: {v:.3f}")
            st.progress(float(v))

        st.markdown("---")
        (st.error if is_drowsy else st.success)(
            f"{'DROWSY' if is_drowsy else 'ALERT'}  •  {detail}"
        )

else:
    st.info("Webcam mode available; keep image mode for debugging first.")
