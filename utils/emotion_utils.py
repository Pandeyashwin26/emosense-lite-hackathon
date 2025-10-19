import os
from datetime import datetime
from typing import Any

# Optional imports with safe fallbacks. Use '# type: ignore' to quiet static analyzers about missing packages.
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    import numpy as np  # type: ignore
except Exception:
    np = None

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

try:
    from transformers import pipeline  # type: ignore
except Exception:
    # Minimal fallback pipeline returning neutral sentiment
    def pipeline(task, model=None):
        def _pipe(texts):
            return [{"label": "NEUTRAL", "score": 0.5}]
        return _pipe

# Optional: audio classification pipeline (Hugging Face) - loaded lazily
_audio_pipeline = None
_audio_model_name = os.environ.get("EMOSENSE_AUDIO_MODEL", "superb/hubert-large-superb-er")

try:
    import librosa  # type: ignore
except Exception:
    librosa = None

try:
    from fer import FER  # type: ignore
except Exception:
    FER = None

# Map model labels to emotion indices (0-4): happy, sad, angry, neutral, surprised
EMOTIONS = ["happy", "sad", "angry", "neutral", "surprised"]

_sentiment = None
_fer_detector = None


def init_text_pipeline():
    global _sentiment
    if _sentiment is None:
        _sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return _sentiment


def init_fer_detector():
    global _fer_detector
    if _fer_detector is None:
        if FER is None:
            _fer_detector = None
        else:
            _fer_detector = FER(mtcnn=True)
    return _fer_detector


def analyze_text_sentiment(text: str) -> dict:
    """
    Returns a dict of emotion probabilities for text mapped to EMOTIONS ordering.
    We map positive -> happy, negative -> sad, neutral/other -> neutral.
    """
    pipe = init_text_pipeline()
    out = pipe(text[:512])
    # out example: [{'label': 'POSITIVE', 'score': 0.9998}]
    label = out[0]["label"]
    score = float(out[0]["score"])
    if np is not None:
        probs = np.zeros(len(EMOTIONS), dtype=float)
        if label == "POSITIVE":
            probs[0] = score
            probs[3] = 1 - score
        else:
            probs[1] = score
            probs[3] = 1 - score
        return {"probs": probs.tolist(), "label": label, "score": score}
    else:
        # fallback to simple list
        probs = [0.0] * len(EMOTIONS)
        if label == "POSITIVE":
            probs[0] = score
            probs[3] = 1 - score
        else:
            probs[1] = score
            probs[3] = 1 - score
        return {"probs": probs, "label": label, "score": score}


def analyze_facial_emotion(frame: Any) -> dict:
    """
    frame: BGR image from OpenCV
    returns prob vector over EMOTIONS
    Uses fer.FER detector to output emotion probabilities; we map labels.
    """
    detector = init_fer_detector()
    # If we don't have detector or cv2, return neutral
    if detector is None or cv2 is None:
        probs = [0.0] * len(EMOTIONS)
        probs[3] = 1.0
        return {"probs": probs, "label": None}

    # FER expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.top_emotion(rgb)
    # top_emotion returns (label, score) or (None, 0.0)
    probs = np.zeros(len(EMOTIONS), dtype=float)
    if results is None:
        return {"probs": probs.tolist(), "label": None}
    label, score = results
    if label is None:
        return {"probs": probs.tolist(), "label": None}
    label = label.lower()
    # map FER labels (e.g., 'happy', 'sad', 'surprise', 'angry', 'neutral', 'fear', 'disgust')
    if "surprise" in label or label == "surprised":
        probs[4] = score
    elif "happy" in label:
        probs[0] = score
    elif "sad" in label:
        probs[1] = score
    elif "angry" in label:
        probs[2] = score
    elif "neutral" in label:
        probs[3] = score
    else:
        # assign to neutral fallback
        probs[3] = score
    # normalize
    s = probs.sum()
    if s > 0:
        probs = probs / s
    return {"probs": probs.tolist(), "label": label, "score": float(score)}


def analyze_audio_emotion(wav_path: str) -> dict:
    """
    Loads a wav file, extracts MFCCs and uses a simple heuristic to predict emotion.
    For prototype purposes, we'll use energy and pitch heuristics to map to emotions.
    """
    # Try to run an HF audio-classification pipeline if available; otherwise use heuristic
    global _audio_pipeline
    if _audio_pipeline is None:
        try:
            _audio_pipeline = pipeline("audio-classification", model=_audio_model_name)
        except Exception:
            _audio_pipeline = None

    if _audio_pipeline is not None:
        try:
            hf_out = _audio_pipeline(wav_path)
            # hf_out can be list of {'label':..., 'score':...}
            # Map common emotion labels to our EMOTIONS vector heuristically
            probs = [0.0] * len(EMOTIONS)
            for item in hf_out:
                label = item.get("label", "").lower()
                score = float(item.get("score", 0.0))
                if "happy" in label or "joy" in label or "positive" in label:
                    probs[0] += score
                elif "sad" in label or "sadness" in label or "negative" in label:
                    probs[1] += score
                elif "angry" in label or "anger" in label:
                    probs[2] += score
                elif "surprise" in label or "surpr" in label:
                    probs[4] += score
                else:
                    probs[3] += score
            s = sum(probs)
            if s > 0:
                probs = [p / s for p in probs]
            return {"probs": probs, "features": {"source": "hf_audio"}}
        except Exception:
            # fallback to heuristic if HF pipeline fails
            _audio_pipeline = None

    if librosa is None:
        # fallback neutral
        probs = [0.0] * len(EMOTIONS)
        probs[3] = 1.0
        return {"probs": probs, "features": {"energy": 0.0, "zcr": 0.0}}

    try:
        y, sr = librosa.load(wav_path, sr=16000)
    except Exception:
        # Handle missing or invalid audio files
        probs = [0.0] * len(EMOTIONS)
        probs[3] = 1.0
        return {"probs": probs, "features": {"error": "file_not_found"}}
    # Extract MFCC mean
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    # Simple heuristics: energy and zero-crossing rate
    energy = np.sqrt(np.mean(y ** 2))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    probs = np.zeros(len(EMOTIONS), dtype=float)
    # Heuristic mapping (toy): high energy & high zcr -> angry or happy; low energy -> sad; medium -> neutral
    if energy > 0.03 and zcr > 0.1:
        probs[2] = 0.5  # angry
        probs[0] = 0.4  # happy
        probs[3] = 0.1
    elif energy > 0.02:
        probs[0] = 0.6
        probs[4] = 0.2
        probs[3] = 0.2
    else:
        probs[1] = 0.7
        probs[3] = 0.3
    # normalize
    probs = probs / probs.sum()
    return {"probs": probs.tolist(), "features": {"energy": float(energy), "zcr": float(zcr)}}


def fuse_modalities(facial_probs, audio_probs, text_probs, weights=(0.4, 0.3, 0.3)):
    """
    Each probs arg is a list-like of length len(EMOTIONS)
    Returns fused probabilities and chosen label
    """
    w_f, w_a, w_t = weights
    # If numpy is available, use it for vector math
    if np is not None:
        f = np.array(facial_probs)
        a = np.array(audio_probs)
        t = np.array(text_probs)
        fused = w_f * f + w_a * a + w_t * t
        # normalize
        if fused.sum() > 0:
            fused = fused / fused.sum()
        idx = int(np.argmax(fused))
        return {"probs": fused.tolist(), "label": EMOTIONS[idx], "index": idx}
    else:
        # pure python fallback
        fused = [0.0] * len(EMOTIONS)
        for i in range(len(EMOTIONS)):
            fused[i] = w_f * facial_probs[i] + w_a * audio_probs[i] + w_t * text_probs[i]
        s = sum(fused)
        if s > 0:
            fused = [x / s for x in fused]
        idx = int(max(range(len(fused)), key=lambda i: fused[i]))
        return {"probs": fused, "label": EMOTIONS[idx], "index": idx}


def log_to_csv(row: dict, csv_path: str = "emotions_log.csv"):
    # prefer pandas if available, otherwise append CSV manually
    if pd is not None:
        df = pd.DataFrame([row])
        header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode="a", header=header, index=False)
    else:
        import csv
        header = not os.path.exists(csv_path)
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if header:
                writer.writeheader()
            writer.writerow(row)


def map_label_to_score(label: str) -> float:
    """Map emotion label to a numeric mood score [-1,1] where negative is bad."""
    if label == "happy":
        return 1.0
    if label == "surprised":
        return 0.5
    if label == "neutral":
        return 0.0
    if label == "sad":
        return -0.7
    if label == "angry":
        return -1.0
    return 0.0
