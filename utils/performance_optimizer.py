"""
Performance optimization utilities for real-time emotion analysis
"""

import time
import threading
import queue
from functools import lru_cache
from typing import Dict, Any, Optional
import numpy as np

class PerformanceOptimizer:
    """
    Optimizes performance for real-time emotion analysis
    """
    
    def __init__(self):
        self.cache = {}
        self.processing_queue = queue.Queue()
        self.result_cache = {}
        self.last_analysis_time = {}
        
        # Performance settings
        self.min_analysis_interval = 0.5  # Minimum seconds between analyses
        self.cache_duration = 30  # Cache results for 30 seconds
        
    @lru_cache(maxsize=100)
    def cached_text_analysis(self, text_hash: str, text: str) -> Dict:
        """Cached text analysis to avoid reprocessing same text"""
        from utils.emotion_utils import analyze_text_sentiment
        return analyze_text_sentiment(text)
    
    def should_analyze(self, modality: str) -> bool:
        """Check if enough time has passed for new analysis"""
        current_time = time.time()
        last_time = self.last_analysis_time.get(modality, 0)
        
        if current_time - last_time >= self.min_analysis_interval:
            self.last_analysis_time[modality] = current_time
            return True
        return False
    
    def get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get cached result if still valid"""
        if cache_key in self.result_cache:
            result, timestamp = self.result_cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                return result
            else:
                # Remove expired cache
                del self.result_cache[cache_key]
        return None
    
    def cache_result(self, cache_key: str, result: Dict):
        """Cache analysis result"""
        self.result_cache[cache_key] = (result, time.time())
    
    def optimize_text_analysis(self, text: str) -> Dict:
        """Optimized text analysis with caching"""
        if not text or len(text.strip()) < 3:
            return {"probs": [0, 0, 0, 1, 0], "label": "neutral", "score": 0.5}
        
        # Create cache key
        text_hash = str(hash(text.strip().lower()))
        cache_key = f"text_{text_hash}"
        
        # Check cache first
        cached = self.get_cached_result(cache_key)
        if cached:
            return cached
        
        # Check if we should analyze (rate limiting)
        if not self.should_analyze("text"):
            # Return last result if available
            if hasattr(self, 'last_text_result'):
                return self.last_text_result
        
        # Perform analysis
        try:
            result = self.cached_text_analysis(text_hash, text)
            self.cache_result(cache_key, result)
            self.last_text_result = result
            return result
        except Exception as e:
            # Fallback result
            return {"probs": [0, 0, 0, 1, 0], "label": "neutral", "score": 0.5, "error": str(e)}
    
    def optimize_audio_analysis(self, audio_path: str) -> Dict:
        """Optimized audio analysis"""
        cache_key = f"audio_{audio_path}_{int(time.time() / 10)}"  # Cache for 10 seconds
        
        # Check cache
        cached = self.get_cached_result(cache_key)
        if cached:
            return cached
        
        # Rate limiting
        if not self.should_analyze("audio"):
            if hasattr(self, 'last_audio_result'):
                return self.last_audio_result
        
        try:
            # Use advanced audio analyzer if available
            try:
                from utils.advanced_audio import analyze_audio_emotion_advanced
                result = analyze_audio_emotion_advanced(audio_path)
            except ImportError:
                from utils.emotion_utils import analyze_audio_emotion
                result = analyze_audio_emotion(audio_path)
            
            self.cache_result(cache_key, result)
            self.last_audio_result = result
            return result
        except Exception as e:
            return {"probs": [0, 0, 0, 1, 0], "label": "neutral", "score": 0.5, "error": str(e)}
    
    def optimize_facial_analysis(self, image_data: Any) -> Dict:
        """Optimized facial analysis"""
        # Rate limiting for facial analysis
        if not self.should_analyze("facial"):
            if hasattr(self, 'last_facial_result'):
                return self.last_facial_result
        
        try:
            from utils.emotion_utils import analyze_facial_emotion
            result = analyze_facial_emotion(image_data)
            self.last_facial_result = result
            return result
        except Exception as e:
            return {"probs": [0, 0, 0, 1, 0], "label": "neutral", "score": 0.5, "error": str(e)}
    
    def fast_fusion(self, face_probs, audio_probs, text_probs, weights=(0.4, 0.3, 0.3)) -> Dict:
        """Optimized fusion calculation"""
        try:
            # Ensure all inputs are lists
            if face_probs is None:
                face_probs = [0, 0, 0, 1, 0]
            if audio_probs is None:
                audio_probs = [0, 0, 0, 1, 0]
            if text_probs is None:
                text_probs = [0, 0, 0, 1, 0]
            
            # Fast numpy-based fusion
            f = np.array(face_probs)
            a = np.array(audio_probs)
            t = np.array(text_probs)
            w = np.array(weights)
            
            # Weighted combination
            fused = w[0] * f + w[1] * a + w[2] * t
            
            # Normalize
            fused = fused / fused.sum()
            
            # Find dominant emotion
            emotions = ["happy", "sad", "angry", "neutral", "surprised"]
            max_idx = int(np.argmax(fused))
            
            return {
                "probs": fused.tolist(),
                "label": emotions[max_idx],
                "index": max_idx,
                "confidence": float(fused[max_idx])
            }
        except Exception as e:
            # Fallback
            return {
                "probs": [0, 0, 0, 1, 0],
                "label": "neutral",
                "index": 3,
                "confidence": 1.0,
                "error": str(e)
            }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            "cache_size": len(self.result_cache),
            "last_analysis_times": self.last_analysis_time.copy(),
            "min_interval": self.min_analysis_interval,
            "cache_duration": self.cache_duration
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.result_cache.clear()
        self.last_analysis_time.clear()
        if hasattr(self, 'last_text_result'):
            delattr(self, 'last_text_result')
        if hasattr(self, 'last_audio_result'):
            delattr(self, 'last_audio_result')
        if hasattr(self, 'last_facial_result'):
            delattr(self, 'last_facial_result')


# Global optimizer instance
_performance_optimizer = None

def get_performance_optimizer():
    """Get or create performance optimizer instance"""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


class RealTimeMLAdapter:
    """
    Simulates real-time ML adaptation for demo purposes
    """
    
    def __init__(self):
        self.user_feedback = []
        self.adaptation_history = []
        self.confidence_threshold = 0.7
        
    def add_user_feedback(self, predicted_emotion: str, actual_emotion: str, confidence: float):
        """Add user feedback for model adaptation"""
        feedback = {
            "predicted": predicted_emotion,
            "actual": actual_emotion,
            "confidence": confidence,
            "timestamp": time.time(),
            "correct": predicted_emotion == actual_emotion
        }
        self.user_feedback.append(feedback)
        
        # Keep only recent feedback (last 100 entries)
        if len(self.user_feedback) > 100:
            self.user_feedback = self.user_feedback[-100:]
    
    def get_adaptation_weights(self) -> Dict[str, float]:
        """Calculate adaptive weights based on user feedback"""
        if len(self.user_feedback) < 5:
            return {"face": 0.4, "audio": 0.3, "text": 0.3}
        
        # Analyze recent feedback
        recent_feedback = self.user_feedback[-20:]  # Last 20 feedbacks
        
        # Calculate accuracy by modality (simulated)
        face_accuracy = 0.8 + np.random.normal(0, 0.05)  # Simulate learning
        audio_accuracy = 0.75 + np.random.normal(0, 0.05)
        text_accuracy = 0.85 + np.random.normal(0, 0.05)
        
        # Normalize accuracies to weights
        total_accuracy = face_accuracy + audio_accuracy + text_accuracy
        
        adaptive_weights = {
            "face": face_accuracy / total_accuracy,
            "audio": audio_accuracy / total_accuracy,
            "text": text_accuracy / total_accuracy
        }
        
        return adaptive_weights
    
    def simulate_online_learning(self, emotion_data: Dict) -> Dict:
        """Simulate online learning adaptation"""
        # Add some noise to simulate learning
        if "probs" in emotion_data:
            probs = np.array(emotion_data["probs"])
            
            # Simulate confidence improvement over time
            learning_factor = min(len(self.user_feedback) / 100.0, 0.1)
            confidence_boost = 1.0 + learning_factor
            
            # Boost the dominant emotion slightly
            max_idx = np.argmax(probs)
            probs[max_idx] *= confidence_boost
            
            # Renormalize
            probs = probs / probs.sum()
            
            emotion_data["probs"] = probs.tolist()
            emotion_data["confidence"] = float(probs[max_idx])
            emotion_data["adapted"] = True
            emotion_data["learning_factor"] = learning_factor
        
        return emotion_data
    
    def get_learning_stats(self) -> Dict:
        """Get learning statistics for display"""
        if not self.user_feedback:
            return {"total_feedback": 0, "accuracy": 0.0, "learning_progress": 0.0}
        
        total_feedback = len(self.user_feedback)
        correct_predictions = sum(1 for f in self.user_feedback if f["correct"])
        accuracy = correct_predictions / total_feedback if total_feedback > 0 else 0.0
        
        # Simulate learning progress
        learning_progress = min(total_feedback / 50.0, 1.0)  # Progress towards 50 feedbacks
        
        return {
            "total_feedback": total_feedback,
            "accuracy": accuracy,
            "learning_progress": learning_progress,
            "recent_accuracy": accuracy  # Could calculate recent vs overall
        }


# Global ML adapter instance
_ml_adapter = None

def get_ml_adapter():
    """Get or create ML adapter instance"""
    global _ml_adapter
    if _ml_adapter is None:
        _ml_adapter = RealTimeMLAdapter()
    return _ml_adapter