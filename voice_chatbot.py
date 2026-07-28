"""
Simple Voice-Based Chatbot
---------------------------
Audio in (mic) -> Speech-to-Text (faster-whisper, offline)
              -> LLM response (via internal LiteLLM gateway)
              -> Text-to-Speech (pyttsx3, offline)      -> Audio out (speaker)

Setup:
    python -m venv voicebot-env
    voicebot-env\\Scripts\\activate        (Windows)
    pip install -r requirements.txt

    Copy .env.example to .env and fill in your LiteLLM gateway details.

Run:
    python voice_chatbot.py
"""

import os
import ssl
import wave
import time

# Corporate proxy does SSL inspection; patch ssl to use system trust store
ssl._create_default_https_context = ssl.create_default_context
try:
    import certifi, httpx
    _ssl_ctx = ssl.create_default_context()
    _ssl_ctx.load_verify_locations(certifi.where())
except Exception:
    pass
os.environ.setdefault("CURL_CA_BUNDLE", "")
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

import numpy as np
import sounddevice as sd
ENERGY_THRESHOLD = 300  # RMS amplitude; raise if mic picks up too much background noise
import pyttsx3
from faster_whisper import WhisperModel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Configuration
# ----------------------------
SAMPLE_RATE = 16000
FRAME_MS = 30  # webrtcvad requires 10, 20, or 30 ms frames
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)
SILENCE_LIMIT_MS = 1200  # how long silence must last to consider speech "done"
MAX_RECORD_SECONDS = 15  # hard cap so it never records forever

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")  # tiny/base/small
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "2"))  # 0-3, higher = more aggressive filtering

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "https://genailab.tcs.in/litellm/v1")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "sk-lbqqCkOWyiJ3WXzgpCj6yQ")
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "gpt-4o-mini")

TTS_RATE = int(os.getenv("TTS_RATE", "175"))

EXIT_WORDS = {"exit", "quit", "stop", "goodbye", "bye"}
SYSTEM_PROMPT = "You are a helpful assistant. Always respond in English language."


# ----------------------------
# Initialization
# ----------------------------
print("Loading speech-to-text model...")
stt_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

print("Connecting to LLM gateway...")
llm = ChatOpenAI(
    model=LITELLM_MODEL,
    base_url=LITELLM_BASE_URL,
    api_key=LITELLM_API_KEY,
    temperature=0.5,
    http_client=httpx.Client(verify=False),
)

print("Initializing text-to-speech engine...")
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", TTS_RATE)



# ----------------------------
# Audio recording with voice-activity detection
# ----------------------------
def record_until_silence(filename="input.wav"):
    """Records from the mic and stops automatically after sustained silence."""
    print("\nListening... (speak now)")

    frames = []
    silence_frames = 0
    silence_frame_limit = int(SILENCE_LIMIT_MS / FRAME_MS)
    max_frames = int(MAX_RECORD_SECONDS * 1000 / FRAME_MS)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME_SIZE
    )

    speech_started = False

    with stream:
        for _ in range(max_frames):
            audio_block, _ = stream.read(FRAME_SIZE)
            frame_bytes = audio_block.tobytes()

            is_speech = np.sqrt(np.mean(audio_block.astype(np.float32) ** 2)) > ENERGY_THRESHOLD
            frames.append(audio_block.copy())

            if is_speech:
                speech_started = True
                silence_frames = 0
            elif speech_started:
                silence_frames += 1
                if silence_frames > silence_frame_limit:
                    break

    if not frames:
        return None

    audio_data = np.concatenate(frames, axis=0)

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())

    return filename


# ----------------------------
# Speech-to-Text
# ----------------------------
def transcribe(filename):
    segments, _ = stt_model.transcribe(filename, language="en")
    text = " ".join(seg.text for seg in segments).strip()
    return text


# ----------------------------
# LLM call
# ----------------------------
def get_response(user_text, history=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + (history or []) + [{"role": "user", "content": user_text}]
    result = llm.invoke(messages)
    return result.content


# ----------------------------
# Text-to-Speech
# ----------------------------
def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()


# ----------------------------
# Main loop
# ----------------------------
def voice_chat_loop():
    conversation_history = []
    print("\nVoice chatbot ready. Say 'exit' or 'stop' to quit.\n")

    while True:
        try:
            audio_file = record_until_silence()
            if not audio_file:
                continue

            user_text = transcribe(audio_file)
            if not user_text:
                print("(no speech detected, try again)")
                continue

            print(f"You said: {user_text}")

            if user_text.strip().lower().strip(".!?") in EXIT_WORDS:
                speak("Goodbye!")
                break

            reply = get_response(user_text, conversation_history)
            print(f"Bot: {reply}")

            conversation_history.append({"role": "user", "content": user_text})
            conversation_history.append({"role": "assistant", "content": reply})

            speak(reply)

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    voice_chat_loop()
