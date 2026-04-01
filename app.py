import streamlit as st
import numpy as np
import librosa
import json
import os
import time
from google import genai
from google.genai import types

st.set_page_config(page_title="Audio Quality-Control Dashboard", layout="wide")

st.title("Audio Quality-Control Dashboard")
st.write("Upload an audio file to run deterministic quality checks and AI-powered semantic analysis.")

api_key = st.text_input("Gemini API Key", type="password", help="Enter your Gemini API key to run the AI analysis.")

uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button("Run Analysis"):
        if not api_key:
            st.error("Please enter your Gemini API Key.")
        else:
            with st.spinner("Analyzing audio and generating AI metrics... This may take a minute."):
                # Save uploaded file temporarily for librosa and Gemini upload
                temp_filename = f"temp_{uploaded_file.name}"
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # --- 1. Audio Processing (Deterministic Metrics) ---
                    y, sr = librosa.load(temp_filename, sr=None, mono=True)
                    duration = librosa.get_duration(y=y, sr=sr)
                    
                    rms = librosa.feature.rms(y=y)[0]
                    clip_count = np.sum(np.abs(y) >= 0.999)
                    
                    signal_rms = np.percentile(rms, 90)
                    noise_rms = np.percentile(rms, 10)
                    snr_db = 20 * np.log10(signal_rms / (noise_rms + 1e-9))
                    
                    silence_threshold = 0.001
                    silence_samples = np.sum(np.abs(y) < silence_threshold)
                    silence_ratio = (silence_samples / len(y)) * 100
                    
                    st.subheader("Deterministic Audio Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Duration", f"{duration:.2f}s")
                    col2.metric("SNR", f"{snr_db:.2f} dB")
                    col3.metric("Clipping", "Detected" if clip_count > 0 else "None")
                    col4.metric("Silence Ratio", f"{silence_ratio:.1f}%")
                    
                    # --- 2. Gemini AI Analysis ---
                    st.subheader("AI Semantic Analysis & Transcription")
                    client = genai.Client(api_key=api_key)
                    
                    # Upload to Gemini
                    gemini_file = client.files.upload(file=temp_filename)
                    
                    prompt = """Analyze the audio and provide a JSON response with the following structure. ALL SCORES MUST BE NUMBERS (e.g. 0.5), NOT STRINGS:
                    {
                      "content_safety_scores": {
                        "toxicity_detected": "Number 0.0 to 1.0",
                        "sexual_content_detected": "Number 0.0 to 1.0",
                        "violent_intent": "Number 0.0 to 1.0",
                        "political_campaigning": "Number 0.0 to 1.0",
                        "discriminatory_content": "Number 0.0 to 1.0",
                        "pii_leakage": "Number (Count of real PII entities)"
                      },
                      "conversation_quality_metrics": {
                        "is_scripted": "Number 0.0 to 1.0",
                        "real_world_impersonation": "Number 0.0 to 1.0",
                        "excessive_code_switching": "Number 0.0 to 1.0",
                        "topic_coherence_fail": "Number 0.0 to 1.0",
                        "native_speaker_fail": "Number 0.0 to 1.0",
                        "task_alignment_fail": "Number 0.0 to 1.0",
                        "emotion_sentiment_mismatch": "Number 0.0 to 1.0"
                      },
                      "voice_quality_metrics": {
                        "unnatural_pauses": "Number 0.0 to 1.0",
                        "robotic_tone": "Number 0.0 to 1.0",
                        "audio_glitches": "Number 0.0 to 1.0"
                      },
                      "speakers": ["List of IDs"],
                      "transcript_by_turn": [
                        {
                          "speaker": "Speaker 1",
                          "start_time": "[MM:SS]",
                          "end_time": "[MM:SS]",
                          "text": "Exact verbatim text"
                        }
                      ]
                    }
                    **TRANSCRIPTION RULES:**
                    - Beeps / Sensitive Info: If a beep replaces PII (name, DOB, phone), write [beep]. Never guess the hidden info.
                    - Overlapping / Interruptions: If two speakers talk at the same time, write each speaker on a separate line with their respective timestamps. Do NOT add notes like (overlapping) or (interruption).
                    - Fillers: Keep natural fillers exactly as spoken (um, uh, ah, like, you know).
                    - Cut-off Sentences: If a speaker is interrupted, end with a dash —.
                    - Accuracy: Write exactly what you hear. No paraphrasing or commentary.
                    - Speaker Labels: Always use Speaker 1, Speaker 2, etc. Each speaker always starts on a new line."""
                    
                    # Retry logic for 503/429 errors
                    max_retries = 4
                    response = None
                    for attempt in range(max_retries):
                        try:
                            response = client.models.generate_content(
                                model='gemini-3.1-pro-preview',
                                contents=[gemini_file, prompt],
                                config=types.GenerateContentConfig(
                                    response_mime_type="application/json",
                                )
                            )
                            break
                        except Exception as e:
                            if "503" in str(e) or "429" in str(e):
                                if attempt < max_retries - 1:
                                    time.sleep(2 ** attempt)
                                    continue
                            raise e
                    
                    if response:
                        try:
                            json_data = json.loads(response.text)
                            st.json(json_data)
                        except json.JSONDecodeError:
                            st.error("Failed to parse JSON from AI response.")
                            st.text(response.text)
                            
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                finally:
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
