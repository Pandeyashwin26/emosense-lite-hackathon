try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    raise

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase  # type: ignore
except Exception:  # pragma: no cover
    webrtc_streamer = None
    VideoTransformerBase = object

import numpy as np  # type: ignore
import soundfile as sf  # type: ignore
import tempfile
import os
from datetime import datetime
from utils.emotion_utils import analyze_facial_emotion, analyze_audio_emotion, analyze_text_sentiment, fuse_modalities, log_to_csv, map_label_to_score

st.set_page_config(page_title="EmoSense Lite", layout="wide")

st.title("EmoSense Lite — Multimodal Mental Health Companion")
# Show logo if available
try:
    import base64
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo_simple.svg")
    if os.path.exists(logo_path):
        with open(logo_path, "r", encoding="utf-8") as f:
            svg = f.read()
        b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
        st.markdown(f"<img src=\"data:image/svg+xml;base64,{b64}\" width=120/>", unsafe_allow_html=True)
except Exception:
    pass

# Sidebar for simple controls
with st.sidebar:
    st.header("Settings")
    weights_face = st.slider("Face weight", 0.0, 1.0, 0.4, 0.05)
    weights_voice = st.slider("Voice weight", 0.0, 1.0, 0.3, 0.05)
    weights_text = st.slider("Text weight", 0.0, 1.0, 0.3, 0.05)
    if abs(weights_face + weights_voice + weights_text - 1.0) > 0.01:
        st.info("Weights will be normalized to sum to 1")

# Tabs
tabs = st.tabs(["Face Cam", "Voice Test", "Text Journal", "Dashboard"]) 

# Shared state
if "last_face" not in st.session_state:
    st.session_state["last_face"] = None
if "last_audio" not in st.session_state:
    st.session_state["last_audio"] = None
if "last_text" not in st.session_state:
    st.session_state["last_text"] = None
if "last_fused" not in st.session_state:
    st.session_state["last_fused"] = None

# Face Cam Tab
with tabs[0]:
    st.header("Face Cam — Live emotion (frames)")
    st.write("Allow camera access; the app will analyze your face and show the detected emotion.")

    if webrtc_streamer is not None and cv2 is not None:
        class EmotionTransformer(VideoTransformerBase):
            def __init__(self):
                self.frame = None
            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                res = analyze_facial_emotion(img)
                st.session_state["last_face"] = res
                # overlay label
                label = res.get("label") or ""
                score = res.get("score", 0)
                if cv2 is not None:
                    cv2.putText(img, f"{label} ({score:.2f})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
                return img

        webrtc_ctx = webrtc_streamer(key="emo-cam", video_transformer_factory=EmotionTransformer, rtc_configuration={})
    else:
        st.error("Camera functionality requires opencv-python and streamlit-webrtc packages.")
        st.info("Install with: pip install opencv-python streamlit-webrtc")

# Voice Test Tab
with tabs[1]:
    st.header("Voice Test — Record 5 seconds")
    st.write("Record a short voice sample (5s). We'll extract MFCCs and give a heuristic emotion prediction.")
    audio_record = st.audio
    # Use file_uploader as a simple recorder placeholder for prototype
    uploaded = st.file_uploader("Upload a .wav file (5s) or record via another app and upload", type=["wav","mp3"], accept_multiple_files=False)
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        res_audio = analyze_audio_emotion(tmp_path)
        st.session_state["last_audio"] = res_audio
        st.write("Audio features:", res_audio.get("features"))
        st.write("Audio probs:", res_audio.get("probs"))

# Text Journal Tab
with tabs[2]:
    st.header("Text Journal")
    st.write("Type your thoughts — the app will analyze sentiment.")
    text = st.text_area("Write here:", height=200)
    if st.button("Analyze text"):
        if text.strip():
            res_text = analyze_text_sentiment(text)
            st.session_state["last_text"] = res_text
            st.write("Text sentiment:", res_text)
        else:
            st.info("Please enter some text to analyze.")

# Dashboard Tab
with tabs[3]:
    st.header("Dashboard")
    st.write("Latest per-modality results and fused overall mood.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Face")
        st.write(st.session_state.get("last_face"))
    with col2:
        st.subheader("Voice")
        st.write(st.session_state.get("last_audio"))
    with col3:
        st.subheader("Text")
        st.write(st.session_state.get("last_text"))

    if st.button("Fuse latest and show overall mood"):
        face = st.session_state.get("last_face")
        audio = st.session_state.get("last_audio")
        text = st.session_state.get("last_text")
        # fallback to neutral probs
        neutral = [0,0,0,1,0]
        f_probs = face.get("probs") if face else neutral
        a_probs = audio.get("probs") if audio else neutral
        t_probs = text.get("probs") if text else neutral
        # normalize weights
        wsum = weights_face + weights_voice + weights_text
        w_face = weights_face/wsum
        w_voice = weights_voice/wsum
        w_text = weights_text/wsum
        fused = fuse_modalities(f_probs, a_probs, t_probs, weights=(w_face, w_voice, w_text))
        st.session_state["last_fused"] = fused
        st.metric("Overall emotion", fused.get("label"))
        # show bar chart
        try:
            import pandas as pd  # type: ignore
            df = pd.DataFrame([fused.get("probs")], columns=["happy","sad","angry","neutral","surprised"])
            st.bar_chart(df.T)
        except Exception:
            # fallback: display raw probs
            st.write("Probs:", fused.get("probs"))
        # Suggestion
        score = map_label_to_score(fused.get("label"))
        if score < -0.2:
            st.warning("It seems you may be feeling down or stressed. Try a 4-4-4 deep breathing exercise or write for 5 minutes.")
        else:
            st.success("You're looking okay — keep going! Here's a motivational quote:\n\"The expert in anything was once a beginner.\"")
        # Log
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "face": str(face),
            "audio": str(audio),
            "text": str(text),
            "fused_label": fused.get("label"),
            "fused_probs": str(fused.get("probs"))
        }
        log_to_csv(row)

    # show log preview
    st.subheader("Recent log")
    try:
        import pandas as pd  # type: ignore
        if os.path.exists("emotions_log.csv"):
            df_log = pd.read_csv("emotions_log.csv")
            st.dataframe(df_log.tail(10))
        else:
            st.info("No log yet. Fuse and log to populate data.")
    except Exception:
        # fallback: show raw file tail
        if os.path.exists("emotions_log.csv"):
            with open("emotions_log.csv", "r", encoding="utf-8") as f:
                lines = f.readlines()[-10:]
            st.text("".join(lines))
        else:
            st.info("No log yet. Fuse and log to populate data.")
