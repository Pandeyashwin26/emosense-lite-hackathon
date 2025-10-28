"""
Advanced Audio Emotion Recognition with Multiple Model Support
"""

import numpy as np
import librosa
from typing import Dict, List, Optional
import os

# Emotion mapping for different models
EMOTION_MAPPINGS = {
    "ravdess": {
        "01": "neutral", "02": "neutral", "03": "happy", "04": "sad",
        "05": "angry", "06": "angry", "07": "surprised", "08": "surprised"
    },
    "fer": ["happy", "sad", "angry", "neutral", "surprised"]
}

class AdvancedAudioAnalyzer:
    """
    Advanced audio emotion analyzer with multiple model support
    """
    
    def __init__(self):
        self.models = []
        self.load_models()
    
    def load_models(self):
        """Load multiple audio emotion models"""
        try:
            # Try to load Wav2Vec2 model
            from transformers import pipeline
            self.wav2vec_model = pipeline(
                "audio-classification", 
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
            )
            self.models.append("wav2vec2")
        except Exception:
            self.wav2vec_model = None
        
        try:
            # Try to load alternative model
            from transformers import pipeline
            self.hubert_model = pipeline(
                "audio-classification",
                model="superb/hubert-large-superb-er"
            )
            self.models.append("hubert")
        except Exception:
            self.hubert_model = None
    
    def extract_advanced_features(self, audio_path: str) -> Dict:
        """Extract comprehensive audio features"""
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            
            # Basic features
            energy = np.sqrt(np.mean(y ** 2))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            
            # Chroma features
            chroma = librosa.feature.chroma(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Tempo and rhythm
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Pitch features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            
            return {
                "energy": float(energy),
                "zcr": float(zcr),
                "tempo": float(tempo),
                "pitch_mean": float(pitch_mean),
                "spectral_centroid": float(np.mean(spectral_centroids)),
                "spectral_rolloff": float(np.mean(spectral_rolloff)),
                "spectral_bandwidth": float(np.mean(spectral_bandwidth)),
                "mfcc_mean": mfcc_mean.tolist()[:5],
                "mfcc_std": mfcc_std.tolist()[:5],
                "chroma_mean": chroma_mean.tolist()[:5]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_with_wav2vec2(self, audio_path: str) -> Optional[Dict]:
        """Analyze using Wav2Vec2 model"""
        if not self.wav2vec_model:
            return None
        
        try:
            results = self.wav2vec_model(audio_path)
            emotions = ["happy", "sad", "angry", "neutral", "surprised"]
            probs = [0.0] * len(emotions)
            
            for result in results:
                label = result["label"].lower()
                score = result["score"]
                
                # Enhanced emotion mapping
                if any(word in label for word in ["happy", "joy", "positive", "excited"]):
                    probs[0] += score
                elif any(word in label for word in ["sad", "sadness", "negative", "depressed"]):
                    probs[1] += score
                elif any(word in label for word in ["angry", "anger", "mad", "furious"]):
                    probs[2] += score
                elif any(word in label for word in ["surprise", "surprised", "amazed"]):
                    probs[4] += score
                else:
                    probs[3] += score
            
            # Normalize
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            
            return {
                "probs": probs,
                "model": "wav2vec2",
                "confidence": max(probs)
            }
        except Exception:
            return None
    
    def analyze_with_features(self, features: Dict) -> Dict:
        """Advanced feature-based emotion classification"""
        emotions = ["happy", "sad", "angry", "neutral", "surprised"]
        
        if "error" in features:
            # Return neutral if feature extraction failed
            probs = [0.0, 0.0, 0.0, 1.0, 0.0]
            return {"probs": probs, "model": "fallback"}
        
        # Advanced heuristic classification
        energy = features.get("energy", 0.02)
        tempo = features.get("tempo", 120)
        spectral_centroid = features.get("spectral_centroid", 2000)
        pitch_mean = features.get("pitch_mean", 200)
        zcr = features.get("zcr", 0.05)
        
        # Normalize features
        energy_norm = min(1.0, energy * 25)  # Scale energy
        tempo_norm = tempo / 120.0  # Normalize around 120 BPM
        spectral_norm = spectral_centroid / 3000.0  # Normalize spectral centroid
        pitch_norm = min(1.0, pitch_mean / 300.0)  # Normalize pitch
        zcr_norm = min(1.0, zcr * 20)  # Scale ZCR
        
        # Advanced emotion scoring
        happy_score = (
            energy_norm * 0.3 +
            min(tempo_norm * 1.2, 1.0) * 0.25 +
            spectral_norm * 0.25 +
            pitch_norm * 0.2
        )
        
        sad_score = (
            (1 - energy_norm) * 0.4 +
            (1 - tempo_norm * 0.8) * 0.3 +
            (1 - spectral_norm) * 0.2 +
            (1 - pitch_norm) * 0.1
        )
        
        angry_score = (
            energy_norm * 0.35 +
            zcr_norm * 0.3 +
            spectral_norm * 0.2 +
            abs(tempo_norm - 1.0) * 0.15
        )
        
        surprised_score = (
            energy_norm * 0.25 +
            spectral_norm * 0.3 +
            pitch_norm * 0.25 +
            zcr_norm * 0.2
        )
        
        neutral_score = 1.0 - max(happy_score, sad_score, angry_score, surprised_score)
        neutral_score = max(0.1, neutral_score)  # Ensure minimum neutral probability
        
        # Create probability distribution
        raw_probs = [happy_score, sad_score, angry_score, neutral_score, surprised_score]
        
        # Normalize probabilities
        total = sum(raw_probs)
        probs = [p / total for p in raw_probs]
        
        return {
            "probs": probs,
            "model": "advanced_features",
            "confidence": max(probs),
            "feature_scores": {
                "energy_norm": energy_norm,
                "tempo_norm": tempo_norm,
                "spectral_norm": spectral_norm
            }
        }
    
    def ensemble_prediction(self, predictions: List[Dict]) -> Dict:
        """Combine multiple model predictions"""
        if not predictions:
            return {"probs": [0, 0, 0, 1, 0], "model": "fallback"}
        
        # Weight models by confidence
        weights = []
        all_probs = []
        
        for pred in predictions:
            confidence = pred.get("confidence", 0.5)
            # Higher weight for more confident predictions
            weight = confidence ** 2
            weights.append(weight)
            all_probs.append(pred["probs"])
        
        # Weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return predictions[0]  # Fallback to first prediction
        
        ensemble_probs = [0.0] * 5
        for i in range(5):
            weighted_sum = sum(probs[i] * weight for probs, weight in zip(all_probs, weights))
            ensemble_probs[i] = weighted_sum / total_weight
        
        return {
            "probs": ensemble_probs,
            "model": "ensemble",
            "confidence": max(ensemble_probs),
            "num_models": len(predictions)
        }
    
    def analyze_audio_emotion(self, audio_path: str) -> Dict:
        """Main analysis function with multiple models"""
        predictions = []
        
        # Extract features first
        features = self.extract_advanced_features(audio_path)
        
        # Try Wav2Vec2 model
        wav2vec_result = self.analyze_with_wav2vec2(audio_path)
        if wav2vec_result:
            predictions.append(wav2vec_result)
        
        # Always include feature-based analysis
        feature_result = self.analyze_with_features(features)
        predictions.append(feature_result)
        
        # Ensemble prediction
        final_result = self.ensemble_prediction(predictions)
        
        # Add features to result
        final_result["features"] = features
        final_result["models_used"] = [p.get("model") for p in predictions]
        
        # Determine final emotion label
        emotions = ["happy", "sad", "angry", "neutral", "surprised"]
        max_idx = final_result["probs"].index(max(final_result["probs"]))
        final_result["label"] = emotions[max_idx]
        final_result["score"] = final_result["probs"][max_idx]
        
        return final_result


# Global analyzer instance
_audio_analyzer = None

def get_audio_analyzer():
    """Get or create audio analyzer instance"""
    global _audio_analyzer
    if _audio_analyzer is None:
        _audio_analyzer = AdvancedAudioAnalyzer()
    return _audio_analyzer

def analyze_audio_emotion_advanced(audio_path: str) -> Dict:
    """Advanced audio emotion analysis function"""
    analyzer = get_audio_analyzer()
    return analyzer.analyze_audio_emotion(audio_path)