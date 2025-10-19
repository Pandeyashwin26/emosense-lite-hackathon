# EmoSense Lite — Multimodal Mental Health Companion (Prototype)

## About

This project is developed for the **Multi Modal AI Hackathon** organized by **Indian Institute of Technology (IIT), Mandi**.

**Team Name:** DevDash

**Team Leader:** Ashwin Pandey (participating in the hackathon)

## Overview

This is a lightweight prototype that demonstrates multimodal emotion detection using webcam (face), microphone (audio), and text input. It provides a Streamlit UI with tabs for capturing each modality, fuses the results, and displays feedback and logging.

Features

- Facial emotion detection using `fer` (wraps common CNN models like FER2013)
- Audio emotion heuristic using `librosa` features (MFCC, energy)
- Text sentiment using Hugging Face `distilbert-base-uncased-finetuned-sst-2-english`
- Simple weighted fusion of modalities
- Streamlit dashboard with logging to `emotions_log.csv`

Quick start (Windows PowerShell)

1. Create and activate a virtual environment (optional but recommended)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

1. Install dependencies

```powershell
pip install -r requirements.txt
```

1. Run the app

```powershell
streamlit run app.py
```

Notes

- This is a prototype. Audio model is heuristic; replace with a trained classifier for production.
- The `fer` package uses pretrained models for facial emotion recognition. Results vary by camera and lighting.
- For adaptive feedback, integrate an LLM (OpenAI/Gemini) and keep user privacy in mind.

Stretch

- `sample_data/smartwatch.json` contains example health data for the stretch goal.

## Architecture and Design

High-level overview

- Frontend: Streamlit app (`app.py`) provides tabs for Face Cam, Voice Test, Text Journal, and Dashboard. It handles user interaction, displays visualizations, and calls into local analysis utilities.
- Utilities: `utils/emotion_utils.py` contains the modality analyzers (facial, audio, text), fusion logic, and CSV logging. The utilities are intentionally modular so models can be swapped with minimal changes.
- Models: The prototype relies on lightweight pre-trained components:
  - Facial: `fer` package wrapping a FER2013-style CNN.
  - Text: Hugging Face `distilbert-base-uncased-finetuned-sst-2-english` sentiment pipeline.
  - Audio: heuristic MFCC/energy-based classifier in this prototype; replaceable with a trained speech-emotion classifier (RAVDESS-trained or Hugging Face model).
- Persistence: Local CSV log (`emotions_log.csv`) for trend analysis. For production, replace with a secure database (Postgres, DynamoDB).
- Optional integrations: LLM-based feedback (OpenAI/Gemini), smartwatch data ingestion, or background monitoring via an agent.

Component responsibilities

- `app.py`: UI, session state, invoking modality analysis, fusing results, providing feedback and logging.
- `utils/emotion_utils.py`: encapsulates emotion detection per modality and fusion rules; provides a stable function contract (inputs/outputs) so models can be swapped.
- Models and 3rd-party libs: provide predictions. Wrap them behind small adapters so replacements are low-risk.

Architecture diagram (textual)

User -> Streamlit UI (app.py)

  -> Face frames -> FER model -> facial probs
  -> Upload audio -> librosa or model -> audio probs
  -> Text input -> transformer pipeline -> text probs
  -> Fusion logic -> overall mood
  -> Feedback engine -> display & log

Design notes

- Modularity: keep model adapters small, with the same input/output shapes: list[float] probs mapped to [happy, sad, angry, neutral, surprised].
- Privacy-first: all processing is local by default; do not send audio/video/text to external services unless explicitly configured.
- Extensibility: new modalities (smartwatch, typing patterns) should provide the same probs vector and plug into fusion.

## Use Cases

- Student self-check: quick mood check between study sessions.
- Instructor dashboards (aggregate, anonymous): class-level wellbeing trends (requires explicit anonymization and consent).
- Research prototyping: collect labeled multimodal data for model development and validation.
- Personal journaling: pair text journaling with automated sentiment/context-aware prompts.

## Why this matters (Importance)

- Early detection: frequent lightweight checks can reveal downward trends before crises.
- Low friction: webcam/short audio and a quick text entry is non-intrusive for students.
- Multimodal signals: combining face, voice, and text improves robustness versus a single channel.
- Privacy-preserving baseline: local-only processing respects user data.

## Branding, Logo & Visual Identity Guidance

- Project name: "EmoSense Lite" — clear, approachable, and student-focused.
- Tone: supportive, non-judgmental, encouraging.
- Logo ideas:

  - Symbol: a soft circular gauge or small combination of a speech bubble + heart + camera lens to represent multimodality.
  - Colors: calming palette (soft blue #4A90E2, teal #2BBBAD, neutral gray #6E6E6E). Avoid harsh reds as primary colors.
  - Typography: rounded sans-serif for a friendly feel (e.g., Inter, Poppins).
- Accessibility: ensure color contrast for text and charts, provide alternative text, and keyboard navigation for core flows.

## Privacy, Safety and Ethics

- Consent: always require explicit user consent before enabling camera or microphone. Log consent events separately.
- Data minimization: store only aggregated or pseudonymized data unless explicit opt-in is given.
- No clinical claims: the prototype is not a medical device. Provide clear disclaimers and links to institutional resources when recommending help.
- Security: store CSVs and any persisted data in user-specific private directories; encrypt sensitive logs if used in production.

## Deployment Notes

- Local prototype: run with a Python venv and Streamlit (steps in Quick Start).
- Server deployment: use a production WSGI or container image. When deploying externally:

  - Move to a server-run worker architecture where heavy model inference runs in background workers.
  - Use HTTPS, authentication, and role-based access if serving multiple users.
  - Consider GPU instances for heavier models (audio or vision) but prefer CPU-optimized models for cost-efficiency.

## Testing and Validation

- Unit tests: test fusion logic (`fuse_modalities`) with deterministic vectors and edge cases (missing modalities, zeros).
- Integration tests: simulate webcam frames and small audio files with deterministic outputs from small mock detectors.
- Performance: measure latency for each modality; ensure UI remains responsive (<300ms for per-frame analysis or run analysis every Nth frame).

## Troubleshooting

- ERR_CONNECTION_REFUSED: server not running. Ensure Streamlit started correctly and use the venv's `streamlit.exe` if using a project venv.
- ModuleNotFoundError: ensure packages are installed in the active venv (`pip show <pkg>`), and start Streamlit from the venv.
- Camera issues: browser permission required; close other apps using the camera.

## Next steps (recommended)

1. Replace audio heuristic with an actual speech-emotion model (Hugging Face / RAVDESS) and add tests.
2. Add end-to-end tests and a small CI pipeline running unit tests.
3. Add optional LLM feedback integration (OpenAI/Gemini) behind an opt-in toggle and server-side key storage.

## Contact and Contribution

Contributions welcome. Open issues for bugs and feature requests, and send pull requests with focused changes. Keep changes small and include tests where possible.
