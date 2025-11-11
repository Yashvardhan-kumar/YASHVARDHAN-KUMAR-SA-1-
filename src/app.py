import streamlit as st
import numpy as np
from PIL import Image
import cv2
import mediapipe as mp

# ===== OPTIONAL: your keras model =====
USE_MODEL = True
MODEL_PATH = "models/drowsiness_model.h5"

if USE_MODEL:
    from tensorflow.keras.models import load_model
    @st.cache_resource
    def load_dnn():
        return load_model(MODEL_PATH)
    model = load_dnn()
else:
    model = None

st.set_page_config(page_title="Driver Drowsiness Detector", page_icon="😴", layout="centered")
st.title("🚗 Driver Drowsiness, Eyes & Mouth Detection")

# ---------------- Sidebar thresholds ----------------
st.sidebar.header("Thresholds / Options")
eye_thresh   = st.sidebar.slider("Eyes closed threshold (EAR ↓)", 0.15, 0.35, 0.23, 0.01)
mouth_thresh = st.sidebar.slider("Mouth open threshold (MAR ↑)", 0.40, 0.85, 0.60, 0.01)
model_thresh = st.sidebar.slider("Model ‘drowsy’ threshold", 0.50, 0.99, 0.70, 0.01)
use_webcam   = st.sidebar.checkbox("Use Webcam (else Upload Image)", value=False)

st.sidebar.caption("Tip: EAR lower → eyes closed, MAR higher → mouth open/yawn.")

# -------------- FaceMesh setup -----------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
LEFT_EYE  = [33, 159, 133, 145, 153, 157]  # landmarks sampling eye (approx)
RIGHT_EYE = [362, 386, 263, 374, 380, 390]
MOUTH     = [61, 291, 13, 14, 87, 317]     # corners + top/bottom inner lips

def _dist(a, b):
    return np.linalg.norm(a-b)

def eye_aspect_ratio(land):
    # simple EAR using vertical distances / horizontal distance (avg both eyes)
    def ear_one(ids):
        p = np.array([land[i] for i in ids])
        # horiz approx: corner-to-corner (0 and 2)
        horiz = _dist(p[0], p[2]) + 1e-6
        # verticals: (1-5) and (3-4) pairs
        vert = (_dist(p[1], p[5]) + _dist(p[3], p[4])) / 2.0
        return vert / horiz
    return (ear_one(LEFT_EYE) + ear_one(RIGHT_EYE)) / 2.0

def mouth_aspect_ratio(land):
    # MAR = vertical opening / mouth width
    # corners: 0,1; vertical: 2(top)-3(bottom) and 4-5 helpers
    p = np.array([land[i] for i in MOUTH])
    width = _dist(p[0], p[1]) + 1e-6
    vertical = ( _dist(p[2], p[3]) + _dist(p[4], p[5]) ) / 2.0
    return vertical / width

def run_facemesh(pil):
    img = np.array(pil.convert("RGB"))
    h, w = img.shape[:2]
    res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not res.multi_face_landmarks:
        return None, None, None
    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([[l.x*w, l.y*h] for l in lm], dtype=np.float32)

    # compute metrics
    EAR = eye_aspect_ratio(pts)
    MAR = mouth_aspect_ratio(pts)

    # annotate flags
    eyes_closed  = EAR < eye_thresh
    mouth_open   = MAR > mouth_thresh
    return EAR, MAR, (eyes_closed, mouth_open)

def run_model(pil):
    if model is None:
        return None, {}
    im = pil.resize((224, 224))
    arr = np.array(im).astype("float32")/255.0
    x = np.expand_dims(arr, axis=0)
    y = model.predict(x, verbose=0)[0]
    # normalize to probs
    if y.ndim == 0: y = np.array([y])
    if len(y) == 1:
        # binary sigmoid (unknown scale) → squash then prob
        p = float(1/(1+np.exp(-float(y[0])))) if not (0<=y[0]<=1) else float(y[0])
        return p, {"drowsy_prob": p}
    # softmax safety
    y = y.astype("float64")
    if not (0.98 <= y.sum() <= 1.02):
        y = np.exp(y - y.max()); y = y / y.sum()
    # assume two- or four-class; map best effort
    names = ["Closed","Open","no_yawn","yawn"][:len(y)]
    probs = {names[i]: float(y[i]) for i in range(len(y))}
    p_drowsy = max(probs.get("Closed",0), probs.get("yawn",0), probs.get("drowsy",0))
    return p_drowsy, probs

def decide(ear, mar, eyes_closed, mouth_open, p_model):
    # final rule: if any condition is strong → Drowsy
    votes = []
    if eyes_closed: votes.append("eyes")
    if mouth_open:  votes.append("mouth")
    if p_model is not None and p_model >= model_thresh: votes.append("model")
    return (len(votes) > 0), votes

def show_metrics(ear, mar, probs):
    st.markdown("### Metrics")
    if ear is not None:
        st.write(f"**EAR (eyes)**: {ear:.3f}  — *(< {eye_thresh:.2f} → closed)*")
        st.progress(float(min(max((0.4 - ear)/0.4, 0), 1)))  # visualization only
        st.write(f"**MAR (mouth)**: {mar:.3f} — *(> {mouth_thresh:.2f} → open)*")
        st.progress(float(min(max(mar/1.2, 0), 1)))
    if probs:
        st.write("**Model probabilities:**")
        for k,v in probs.items():
            st.write(f"- {k}: {v:.2f}")
            st.progress(float(v))

# -------- input handling --------
def process_pil(pil):
    ear, mar, flags = run_facemesh(pil)
    probs = {}
    p_model = None
    if model is not None:
        p_model, probs = run_model(pil)
    if flags is None:
        st.warning("No face detected. Try a front-facing, well-lit image.")
        return
    eyes_closed, mouth_open = flags
    is_drowsy, votes = decide(ear, mar, eyes_closed, mouth_open, p_model)
    # labels
    st.markdown("### Status")
    st.write(f"**Eyes:** {'Closed' if eyes_closed else 'Open'}")
    st.write(f"**Mouth:** {'Open' if mouth_open else 'Closed'}")
    if p_model is not None:
        st.write(f"**Model drowsy score:** {p_model:.2f} (threshold {model_thresh:.2f})")

    show_metrics(ear, mar, probs)
    st.markdown("---")
    (st.error if is_drowsy else st.success)(
        f"{'DROWSY' if is_drowsy else 'ALERT'}  •  votes: {', '.join(votes) if votes else 'none'}"
    )

# ---------- UI ----------
if not use_webcam:
    file = st.file_uploader("Upload driver's face image", type=["jpg","jpeg","png"])
    if file:
        pil = Image.open(file).convert("RGB")
        st.image(pil, caption="Input", use_container_width=True)
        process_pil(pil)
else:
    st.info("Webcam mode: press Start")
    run = st.checkbox("Start")
    FRAME = st.image([])
    cap = cv2.VideoCapture(0)
    while run:
        ok, frame = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        ear, mar, flags = run_facemesh(pil)
        if flags is not None:
            eyes_closed, mouth_open = flags
            p_model = None
            if model is not None:
                p_model, _ = run_model(pil)
            is_drowsy, votes = decide(ear, mar, eyes_closed, mouth_open, p_model)
            text = f"{'DROWSY' if is_drowsy else 'ALERT'} | EAR {ear:.2f} | MAR {mar:.2f}"
            color = (255,0,0) if is_drowsy else (0,200,0)
            cv2.putText(rgb, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        FRAME.image(rgb)
    cap.release()
