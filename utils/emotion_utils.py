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
    Enhanced audio emotion detection using trained models and advanced features.
    """
    global _audio_pipeline
    if _audio_pipeline is None:
        try:
            _audio_pipeline = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        except Exception:
            _audio_pipeline = None

    if _audio_pipeline is not None:
        try:
            hf_out = _audio_pipeline(wav_path)
            probs = [0.0] * len(EMOTIONS)
            for item in hf_out:
                label = item.get("label", "").lower()
                score = float(item.get("score", 0.0))
                if any(word in label for word in ["happy", "joy", "positive"]):
                    probs[0] += score
                elif any(word in label for word in ["sad", "sadness", "negative"]):
                    probs[1] += score
                elif any(word in label for word in ["angry", "anger", "mad"]):
                    probs[2] += score
                elif any(word in label for word in ["surprise", "surprised"]):
                    probs[4] += score
                else:
                    probs[3] += score
            s = sum(probs)
            if s > 0:
                probs = [p / s for p in probs]
            return {"probs": probs, "features": {"source": "wav2vec2_model"}}
        except Exception:
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
    # Enhanced feature extraction
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # Additional features for better emotion detection
    energy = np.sqrt(np.mean(y ** 2))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroids)
    
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    except:
        tempo = 120.0
    
    probs = np.zeros(len(EMOTIONS), dtype=float)
    
    # Enhanced emotion classification
    tempo_normalized = tempo / 120.0
    
    # Happy: High energy, moderate-high tempo, high spectral centroid
    happy_score = min(1.0, (energy * 8) * min(tempo_normalized, 1.5) * (spectral_centroid_mean / 2000))
    
    # Sad: Low energy, low tempo, low spectral features
    sad_score = min(1.0, (1 - energy * 4) * (1 - tempo_normalized * 0.7))
    
    # Angry: High energy, high ZCR, variable tempo
    angry_score = min(1.0, (energy * 10) * (zcr * 8))
    
    # Surprised: High energy, high spectral centroid
    surprised_score = min(1.0, (energy * 6) * (spectral_centroid_mean / 3000))
    
    # Neutral: Moderate values
    neutral_score = max(0.1, 1.0 - max(happy_score, sad_score, angry_score, surprised_score))
    
    probs[0] = happy_score
    probs[1] = sad_score
    probs[2] = angry_score
    probs[3] = neutral_score
    probs[4] = surprised_score
    
    # Normalize
    probs = probs / probs.sum()
    
    features = {
        "energy": float(energy),
        "zcr": float(zcr),
        "tempo": float(tempo),
        "spectral_centroid": float(spectral_centroid_mean),
        "source": "enhanced_mfcc"
    }
    
    return {"probs": probs.tolist(), "features": features}


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
