import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Driver Drowsiness Detector 🚗",
    page_icon="😴",
    layout="centered"
)

# ------------------ HEADER SECTION ------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 40px;
            font-weight: bold;
            color: #2E86C1;
            text-align: center;
        }
        .sub-title {
            text-align: center;
            font-size: 18px;
            color: gray;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<p class="main-title">🚗 Driver Drowsiness & Yawn Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Detect driver fatigue from images or live webcam feed</p>', unsafe_allow_html=True)
st.divider()

# ------------------ MODEL LOADING ------------------
@st.cache_resource
def load_drowsiness_model():
    return load_model("models/drowsiness_model.h5")

model = load_drowsiness_model()
classes = ['Closed', 'Open', 'no_yawn', 'yawn']

# ------------------ SIDEBAR ------------------
st.sidebar.header("🧭 Navigation")
option = st.sidebar.radio("Choose Input Type:", ["Upload Image", "Use Webcam"])

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: For best accuracy, ensure clear lighting and face visibility.")

# ------------------ IMAGE UPLOAD SECTION ------------------
if option == "Upload Image":
    st.subheader("📁 Upload Driver's Image")
    file = st.file_uploader("Upload a face image:", type=["jpg", "jpeg", "png"])

    if file:
        img = Image.open(file).convert("RGB").resize((224, 224))
        st.image(img, caption="Uploaded Image", use_container_width=True)
        st.write("🔍 Analyzing the image...")

        x = np.expand_dims(np.array(img) / 255.0, axis=0)
        preds = model.predict(x)[0]
        label = classes[np.argmax(preds)]
        conf = np.max(preds)

        # Styled result box
        st.markdown("---")
        st.markdown(f"### 🧠 **Prediction:** `{label}`")
        st.progress(float(conf))
        st.write(f"**Confidence:** {conf:.2f}")

        if label in ["Closed", "yawn"]:
            st.error("⚠️ **Driver seems DROWSY!** Please take a short break 💤")
        else:
            st.success("✅ **Driver seems ALERT and ATTENTIVE.** Keep driving safely!")

# ------------------ WEBCAM SECTION ------------------
elif option == "Use Webcam":
    st.subheader("🎥 Real-time Detection via Webcam")
    st.info("Click the checkbox below to start the webcam stream. Make sure to grant permission!")

    run = st.checkbox("📸 Start Webcam")
    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)
    while run:
        ret, frame = camera.read()
        if not ret:
            st.warning("Failed to access the webcam.")
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img, (224, 224))
        x = np.expand_dims(resized / 255.0, axis=0)
        preds = model.predict(x)[0]
        label = classes[np.argmax(preds)]

        color = (0, 255, 0) if label in ["Open", "no_yawn"] else (255, 0, 0)
        cv2.putText(img, f"{label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        FRAME_WINDOW.image(img)

    camera.release()