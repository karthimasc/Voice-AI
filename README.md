# Simple Voice Chatbot

Audio-in / audio-out chatbot: mic → speech-to-text (faster-whisper, offline)
→ LLM (via your internal LiteLLM gateway) → text-to-speech (pyttsx3, offline)
→ speaker.

## Setup (Windows)

```bash
python -m venv voicebot-env
voicebot-env\Scripts\activate
pip install -r requirements.txt
```

If pip is blocked by your corporate proxy:

```bash
pip install --proxy http://<proxy-host>:<port> -r requirements.txt
```

## Configure

1. Copy `.env.example` to `.env`
2. Set `LITELLM_BASE_URL`, `LITELLM_API_KEY`, and `LITELLM_MODEL` to match
   your internal gateway and the model alias it exposes.

## Run

```bash
python voice_chatbot.py
```

Speak into your mic. It auto-detects when you stop talking (voice activity
detection) instead of using a fixed recording window. Say "exit", "quit",
"stop", or "goodbye" to end the session.

## Notes

- STT and TTS run fully offline/local — no network calls, so they work fine
  behind a restrictive corporate proxy. Only the LLM call goes out through
  your LiteLLM gateway.
- `WHISPER_MODEL_SIZE` options: `tiny` (fastest, least accurate), `base`
  (good default), `small` (better accuracy, slower).
- `VAD_AGGRESSIVENESS` (0-3): higher values filter out more background
  noise but may cut off soft speech. Start at 2.
- If `pyttsx3` sounds robotic and you want better voice quality, swap it
  out for [Piper TTS](https://github.com/rhasspy/piper) — still fully
  offline, just a slightly heavier setup.
- The script keeps a running conversation history in memory for context;
  it resets each time you restart the script.

## Troubleshooting

- **No mic detected**: run `python -c "import sounddevice as sd; print(sd.query_devices())"`
  to list available input devices, then set the default device if needed.
- **LLM call fails**: confirm `LITELLM_BASE_URL` is reachable from your
  machine and that the model alias in `LITELLM_MODEL` is one your gateway
  actually serves.
- **Choppy/cut-off recordings**: lower `VAD_AGGRESSIVENESS` or increase
  `SILENCE_LIMIT_MS` in the script.
