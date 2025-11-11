import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

st.set_page_config(page_title="Driver Drowsiness Detector 🚗", page_icon="😴", layout="centered")
st.title("🚗 Driver Drowsiness & Yawn Detection")
st.caption("If everything looks drowsy, try changing class order / preprocessing / thresholds on the left.")

# ----------------- Load model -----------------
@st.cache_resource
def load_drowsiness_model():
    return load_model("models/drowsiness_model.h5")

model = load_drowsiness_model()

# ----------------- Sidebar controls -----------------
st.sidebar.header("⚙️ Controls")

# 1) Class order (try these if predictions look wrong)
class_orders = {
    "Closed, Open, no_yawn, yawn": ['Closed', 'Open', 'no_yawn', 'yawn'],
    "Open, Closed, no_yawn, yawn": ['Open', 'Closed', 'no_yawn', 'yawn'],
    "Closed, Open, yawn, no_yawn": ['Closed', 'Open', 'yawn', 'no_yawn'],
    "Open, Closed, yawn, no_yawn": ['Open', 'Closed', 'yawn', 'no_yawn'],
}
cls_choice = st.sidebar.selectbox("Model output order", list(class_orders.keys()), index=0)
CLASSES = class_orders[cls_choice]

# 2) Preprocessing mode
prep_choice = st.sidebar.selectbox("Preprocessing", ["0-1 (rescale)", "ImageNet (mean/std)"], index=0)

# 3) Face-crop toggle
use_face_crop = st.sidebar.checkbox("Crop to face before predicting", value=True)

# 4) Which classes count as “Drowsy”
drowsy_classes = st.sidebar.multiselect(
    "Treat these classes as Drowsy", ["Closed", "yawn"], default=["Closed", "yawn"]
)

# 5) Thresholds
eye_thresh = st.sidebar.slider("Closed threshold", 0.0, 1.0, 0.55, 0.01)
yawn_thresh = st.sidebar.slider("Yawn threshold", 0.0, 1.0, 0.60, 0.01)

st.sidebar.info("Tip: If everything is Drowsy, raise thresholds or change class order / preprocessing.")

# ----------------- Helpers -----------------
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def crop_face(pil_img, expand=0.2):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return pil_img
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    H, W = img.shape[:2]
    x0 = max(0, int(x - expand * w)); y0 = max(0, int(y - expand * h))
    x1 = min(W, int(x + w * (1 + expand))); y1 = min(H, int(y + h * (1 + expand)))
    return Image.fromarray(img[y0:y1, x0:x1])

def preprocess(pil_img, size=(224, 224)):
    if use_face_crop:
        pil_img = crop_face(pil_img)
    pil_img = pil_img.resize(size)
    arr = np.array(pil_img).astype("float32")
    if prep_choice.startswith("0-1"):
        arr = arr / 255.0
    else:
        # ImageNet mean/std
        arr = arr / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype="float32")
        std  = np.array([0.229, 0.224, 0.225], dtype="float32")
        arr = (arr - mean) / std
    x = np.expand_dims(arr, axis=0)
    return x, pil_img  # return processed tensor + what we'll display

def softmax_safe(v):
    v = v.astype("float64")
    v = np.exp(v - v.max())
    return v / v.sum()

def predict_probs(x):
    raw = model.predict(x, verbose=0)[0]  # shape (4,)
    # normalize to probabilities if needed
    probs = raw if 0.98 <= raw.sum() <= 1.02 else softmax_safe(raw)
    # map to chosen class order
    mapping = {cls: float(probs[i]) for i, cls in enumerate(CLASSES)}
    return mapping

def decide_drowsy(probs):
    p_closed = probs.get("Closed", 0.0)
    p_yawn = probs.get("yawn", 0.0)
    is_drowsy = False
    if "Closed" in drowsy_classes and p_closed >= eye_thresh:
        is_drowsy = True
    if "yawn" in drowsy_classes and p_yawn >= yawn_thresh:
        is_drowsy = True
    return is_drowsy, p_closed, p_yawn

# ----------------- UI: Upload or Webcam -----------------
mode = st.radio("Choose input", ["Upload Image", "Use Webcam"], horizontal=True)

if mode == "Upload Image":
    file = st.file_uploader("Upload a driver's face image:", type=["jpg","jpeg","png"])
    if file:
        pil = Image.open(file).convert("RGB")
        x, vis_img = preprocess(pil)
        st.image(vis_img, caption="Used for prediction", use_container_width=True)

        probs = predict_probs(x)
        is_drowsy, p_closed, p_yawn = decide_drowsy(probs)

        # show all probs + top-1
        st.markdown("### Class probabilities")
        top = max(probs.items(), key=lambda kv: kv[1])
        st.write(f"Top-1: **{top[0]}** ({top[1]:.2f})   •   Closed: {p_closed:.2f}   •   Yawn: {p_yawn:.2f}")
        for k, v in probs.items():
            st.write(f"**{k}**: {v:.2f}")
            st.progress(float(v))

        st.markdown("---")
        (st.error if is_drowsy else st.success)(
            "DROWSY" if is_drowsy else "ALERT"
        )

else:
    st.info("Webcam mode unchanged; if you need the same controls there, say the word and I’ll add them.")
