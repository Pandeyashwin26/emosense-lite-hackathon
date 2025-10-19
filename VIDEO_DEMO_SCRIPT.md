# Video Demonstration Script (2-5 minutes)
## EmoSense Lite - Multimodal Mental Health Companion

### 🎬 **Scene 1: Introduction (30 seconds)**
**[Show title slide or README on screen]**

*"Hi, I'm Ashwin Pandey, team leader of DevDash, participating in the IIT Mandi Multi Modal AI Hackathon. Today I'm presenting EmoSense Lite - a multimodal mental health companion that combines facial emotion detection, voice analysis, and text sentiment to provide comprehensive emotional state assessment for students."*

### 🏗️ **Scene 2: Architecture Overview (45 seconds)**
**[Show architecture diagram or draw on whiteboard]**

*"Let me show you our system architecture. EmoSense Lite uses three input modalities:*
- *Face detection using webcam and FER CNN models*
- *Audio analysis with MFCC features and librosa*
- *Text sentiment using Hugging Face DistilBERT*

*These feed into our fusion engine that combines weighted probabilities to give an overall emotional state. Everything processes locally for privacy."*

### 💻 **Scene 3: Live Application Demo (2-3 minutes)**
**[Screen recording of Streamlit app]**

#### **Tab 1: Text Journal (30 seconds)**
*"First, let's test text analysis. I'll type: 'I'm feeling really stressed about my upcoming exams and can't focus on studying.'*
**[Type text and click Analyze]**
*As you can see, it correctly identifies negative sentiment and maps it to sadness with high confidence."*

#### **Tab 2: Voice Test (45 seconds)**
*"Next, audio analysis. I'll upload a sample audio file..."*
**[Upload audio file, show results]**
*"The system extracts MFCC features and energy levels to classify emotional state. Here we see the audio features and probability distribution."*

#### **Tab 3: Dashboard - Fusion (60 seconds)**
*"Now the magic happens in our dashboard. Let me demonstrate multimodal fusion..."*
**[Show individual modality results]**
*"We have results from text and audio. I can adjust the weights - currently 40% face, 30% voice, 30% text. When I click 'Fuse latest results'..."*
**[Click fusion button]**
*"Our algorithm combines all modalities and shows the overall emotional state with a confidence score. The system also provides personalized feedback and logs everything for trend analysis."*

#### **Tab 4: Data Visualization (30 seconds)**
*"Finally, we can see historical data and trends over time, helping identify patterns in emotional wellbeing."*

### 🔧 **Scene 4: Technical Highlights (30 seconds)**
**[Show code structure or technical details]**

*"Key technical achievements:*
- *Modular architecture for easy model swapping*
- *Real-time processing under 300ms*
- *Privacy-first local processing*
- *Comprehensive error handling and fallbacks*
- *Ready for production deployment"*

### 🚀 **Scene 5: Impact & Next Steps (30 seconds)**
**[Show use cases or future vision]**

*"EmoSense Lite addresses critical student mental health needs through:*
- *Early detection of emotional distress*
- *Non-intrusive daily check-ins*
- *Data-driven insights for institutions*

*Next steps include enhanced audio models, LLM integration, and smartwatch connectivity. Thank you!"*

---

## 📋 **Recording Checklist:**

### **Before Recording:**
- [ ] Test Streamlit app is running smoothly
- [ ] Prepare sample text inputs
- [ ] Have sample audio files ready
- [ ] Clear desktop/browser for clean recording
- [ ] Test microphone and screen recording software

### **During Recording:**
- [ ] Speak clearly and at moderate pace
- [ ] Show mouse cursor movements clearly
- [ ] Pause briefly between sections
- [ ] Demonstrate actual functionality, not just slides

### **Technical Setup:**
- [ ] Screen resolution: 1920x1080 recommended
- [ ] Recording software: OBS Studio, Camtasia, or similar
- [ ] Audio: Clear microphone, no background noise
- [ ] Browser: Full screen Streamlit app

### **Sample Inputs to Use:**

#### **Text Examples:**
1. *"I'm feeling really excited about my project presentation tomorrow!"* (Positive)
2. *"I'm overwhelmed with assignments and feeling very stressed."* (Negative)
3. *"Today was an okay day, nothing special happened."* (Neutral)

#### **Demo Flow:**
1. Start with architecture explanation
2. Show text analysis with positive example
3. Show audio analysis (upload file)
4. Demonstrate fusion with multiple modalities
5. Show dashboard and logging features
6. Conclude with impact statement

### **Key Points to Emphasize:**
- ✅ **Working prototype** (not just concept)
- ✅ **Multimodal fusion** (unique selling point)
- ✅ **Privacy-first** approach
- ✅ **Student-focused** design
- ✅ **Production-ready** architecture

---

*Total Duration: 4-5 minutes*  
*Format: MP4, 1080p recommended*  
*Upload to: YouTube, Google Drive, or hackathon platform*