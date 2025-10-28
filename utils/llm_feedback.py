"""
LLM-powered feedback system for personalized mental health suggestions.
"""

import os
from typing import Dict, List, Optional

# Simple rule-based feedback system (can be replaced with actual LLM API)
def generate_personalized_feedback(emotion_data: Dict, user_context: Optional[Dict] = None) -> Dict:
    """
    Generate personalized feedback based on emotion analysis and user context.
    
    Args:
        emotion_data: Dict containing fused emotion results
        user_context: Optional user context (time of day, recent patterns, etc.)
    
    Returns:
        Dict with feedback message, suggestions, and resources
    """
    
    label = emotion_data.get("label", "neutral")
    score = max(emotion_data.get("probs", [0.5]))
    confidence = "high" if score > 0.7 else "medium" if score > 0.5 else "low"
    
    # Enhanced feedback based on emotion and confidence
    feedback_db = {
        "happy": {
            "high": {
                "message": "You're radiating positive energy! 🌟 This is wonderful to see.",
                "suggestions": [
                    "Share your positive mood with friends or family",
                    "Use this energy for creative or productive activities",
                    "Practice gratitude journaling to maintain this feeling",
                    "Consider helping others - spreading positivity multiplies it"
                ],
                "resources": ["Gratitude apps", "Creative writing prompts", "Volunteer opportunities"]
            },
            "medium": {
                "message": "You seem to be in a good mood! 😊 Keep nurturing those positive feelings.",
                "suggestions": [
                    "Take a moment to appreciate what's going well",
                    "Engage in activities you enjoy",
                    "Connect with supportive people in your life"
                ],
                "resources": ["Mindfulness apps", "Social connection activities"]
            }
        },
        "sad": {
            "high": {
                "message": "I notice you're feeling quite down. It's okay to feel this way - your emotions are valid. 💙",
                "suggestions": [
                    "Reach out to a trusted friend, family member, or counselor",
                    "Try gentle movement like a short walk or stretching",
                    "Practice the 4-7-8 breathing technique",
                    "Consider professional support if these feelings persist",
                    "Engage in small, manageable self-care activities"
                ],
                "resources": [
                    "Campus counseling services",
                    "Mental health hotlines",
                    "Meditation apps (Headspace, Calm)",
                    "Online therapy platforms"
                ]
            },
            "medium": {
                "message": "You seem to be experiencing some sadness. Remember, it's normal to have ups and downs. 🤗",
                "suggestions": [
                    "Try journaling about your feelings",
                    "Listen to uplifting music or watch something that makes you smile",
                    "Do a small act of kindness for yourself",
                    "Connect with nature, even if just looking out a window"
                ],
                "resources": ["Journaling apps", "Mood tracking tools", "Nature sounds"]
            }
        },
        "angry": {
            "high": {
                "message": "I can sense you're feeling quite frustrated or angry. Let's work on channeling this energy constructively. 🔥",
                "suggestions": [
                    "Take 10 deep breaths before reacting to anything",
                    "Try physical exercise to release tension (pushups, running, etc.)",
                    "Write down what's bothering you without censoring yourself",
                    "Talk to someone you trust about what's causing these feelings",
                    "Consider if there's a specific problem you can address"
                ],
                "resources": [
                    "Anger management techniques",
                    "Campus gym or exercise facilities",
                    "Stress management workshops"
                ]
            },
            "medium": {
                "message": "You seem a bit frustrated. Let's find healthy ways to process these feelings. 💪",
                "suggestions": [
                    "Take a brief break from whatever is causing stress",
                    "Try progressive muscle relaxation",
                    "Channel energy into something productive"
                ],
                "resources": ["Relaxation techniques", "Stress-relief activities"]
            }
        },
        "neutral": {
            "high": {
                "message": "You're in a balanced emotional state - that's actually quite positive! ⚖️",
                "suggestions": [
                    "This is a great time for planning or decision-making",
                    "Consider setting small, achievable goals",
                    "Use this stability to build positive habits"
                ],
                "resources": ["Goal-setting apps", "Habit trackers", "Planning tools"]
            }
        },
        "surprised": {
            "high": {
                "message": "Something unexpected seems to have caught your attention! 😲",
                "suggestions": [
                    "Take a moment to process what you're experiencing",
                    "If it's positive surprise, savor the moment",
                    "If it's concerning, consider talking to someone about it"
                ],
                "resources": ["Mindfulness techniques", "Processing emotions guides"]
            }
        }
    }
    
    # Get appropriate feedback
    emotion_feedback = feedback_db.get(label, feedback_db["neutral"])
    confidence_feedback = emotion_feedback.get(confidence, emotion_feedback.get("medium", emotion_feedback.get("high")))
    
    # Add contextual elements
    current_hour = user_context.get("hour", 12) if user_context else 12
    
    # Time-based additions
    time_context = ""
    if current_hour < 6:
        time_context = " Since it's very late/early, consider getting some rest."
    elif current_hour < 12:
        time_context = " Starting your morning with emotional awareness is great!"
    elif current_hour < 18:
        time_context = " Afternoon check-ins help maintain emotional balance."
    else:
        time_context = " Evening reflection can help process the day's experiences."
    
    # Combine feedback
    final_message = confidence_feedback["message"] + time_context
    
    return {
        "message": final_message,
        "suggestions": confidence_feedback["suggestions"],
        "resources": confidence_feedback.get("resources", []),
        "confidence": confidence,
        "emotion": label,
        "follow_up": "Remember: If you're consistently experiencing difficult emotions, please reach out to campus counseling services or a mental health professional."
    }


def get_trend_insights(emotion_history: List[Dict]) -> Dict:
    """
    Analyze emotion trends over time and provide insights.
    
    Args:
        emotion_history: List of emotion data over time
    
    Returns:
        Dict with trend analysis and recommendations
    """
    
    if len(emotion_history) < 3:
        return {
            "message": "Keep logging your emotions to see patterns and trends over time.",
            "trend": "insufficient_data"
        }
    
    # Simple trend analysis
    recent_emotions = [entry.get("fused_label", "neutral") for entry in emotion_history[-7:]]  # Last 7 entries
    
    # Count emotions
    emotion_counts = {}
    for emotion in recent_emotions:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    dominant_count = emotion_counts[dominant_emotion]
    
    # Trend insights
    if dominant_count >= len(recent_emotions) * 0.6:  # 60% or more
        if dominant_emotion in ["sad", "angry"]:
            return {
                "message": f"I notice you've been feeling {dominant_emotion} frequently lately. This pattern suggests it might be helpful to talk to someone or try some new coping strategies.",
                "trend": "concerning_pattern",
                "recommendation": "Consider reaching out to campus counseling services or trying stress-reduction techniques.",
                "dominant_emotion": dominant_emotion
            }
        elif dominant_emotion == "happy":
            return {
                "message": "You've been consistently positive lately - that's wonderful! Keep doing what's working for you.",
                "trend": "positive_pattern",
                "recommendation": "Continue your current positive practices and consider sharing what's working with others.",
                "dominant_emotion": dominant_emotion
            }
    
    # Mixed emotions (healthy variety)
    return {
        "message": "You're experiencing a healthy variety of emotions. This emotional flexibility is a sign of good mental health.",
        "trend": "balanced_pattern",
        "recommendation": "Keep monitoring your emotional patterns and practicing self-awareness.",
        "emotion_variety": len(emotion_counts)
    }


# Emergency resources
EMERGENCY_RESOURCES = {
    "crisis_hotlines": [
        "National Suicide Prevention Lifeline: 988",
        "Crisis Text Line: Text HOME to 741741",
        "SAMHSA National Helpline: 1-800-662-4357"
    ],
    "campus_resources": [
        "Campus Counseling Center",
        "Student Health Services",
        "Academic Advisor",
        "Resident Advisor (if in dorms)"
    ],
    "immediate_coping": [
        "Call a trusted friend or family member",
        "Practice grounding: 5 things you see, 4 you hear, 3 you touch, 2 you smell, 1 you taste",
        "Take slow, deep breaths for 2 minutes",
        "Go to a safe, comfortable space"
    ]
}


def check_crisis_indicators(emotion_data: Dict, history: List[Dict]) -> Optional[Dict]:
    """
    Check for patterns that might indicate need for immediate support.
    
    Returns crisis alert if concerning patterns detected.
    """
    
    # Simple crisis detection (in real app, this would be more sophisticated)
    recent_sad_angry = 0
    if len(history) >= 5:
        for entry in history[-5:]:
            if entry.get("fused_label") in ["sad", "angry"]:
                recent_sad_angry += 1
    
    current_negative = emotion_data.get("label") in ["sad", "angry"]
    high_confidence_negative = current_negative and max(emotion_data.get("probs", [0])) > 0.8
    
    if recent_sad_angry >= 4 or high_confidence_negative:
        return {
            "alert": True,
            "message": "I'm noticing some concerning patterns in your emotional state. Please consider reaching out for support.",
            "resources": EMERGENCY_RESOURCES,
            "urgency": "high" if recent_sad_angry >= 4 else "medium"
        }
    
    return None