import os
import io
import asyncio
import requests
from flask import Flask, request, send_file, jsonify
import edge_tts

app = Flask(__name__)

# ---------------- CONFIG ----------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# You can change this to any free model on OpenRouter
# Options (as of 2026):
#   "openrouter/free"                        -> auto-router picks any available free model (MOST RELIABLE, recommended)
#   "nvidia/nemotron-3-nano-30b-a3b:free"    -> fast + good quality (specific model, may get deprecated over time)
#   "meta-llama/llama-3.3-70b-instruct:free" -> higher quality, slightly slower
LLM_MODEL = "openrouter/free"

# Simple in-memory conversation history (per device, keyed by device_id)
conversation_memory = {}

# ---------------- HELPERS ----------------

def transcribe_audio(file_bytes, filename="input.wav"):
    """Send audio to Groq Whisper API and return transcribed text.
    No 'language' param passed -> Whisper auto-detects Bangla/English/mixed."""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    files = {"file": (filename, file_bytes, "audio/wav")}
    data = {"model": "whisper-large-v3"}

    resp = requests.post(GROQ_STT_URL, headers=headers, files=files, data=data)
    resp.raise_for_status()
    return resp.json().get("text", "").strip()


def detect_lang_for_tts(text):
    """Very simple check: if text contains Bangla unicode range, use 'bn',
    otherwise fall back to 'en'. gTTS doesn't support true code-mixed audio,
    so we pick whichever script dominates the reply."""
    bangla_chars = sum(1 for ch in text if '\u0980' <= ch <= '\u09FF')
    return "bn" if bangla_chars > len(text) * 0.2 else "en"


def get_llm_reply(user_text, device_id="default", sensor_context=None):
    """Send text to OpenRouter LLM and get a reply."""
    history = conversation_memory.get(device_id, [])

    system_prompt = (
        "You are ARC, a helpful voice assistant running on an ESP8266 smart device. "
        "Keep replies short (1-3 sentences), clear, and conversational, "
        "since they will be spoken aloud and shown on a small LCD screen. "
        "The user may speak in Bangla, English, or mixed Banglish - reply naturally "
        "in whichever language(s) the user used, matching their style. "
        "Prefer replying mostly in one dominant language (Bangla OR English) per response "
        "so the reply can be converted to speech cleanly, but you may mix a few words if natural."
    )
    if sensor_context:
        system_prompt += f" Current sensor readings: {sensor_context}."

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-6:])  # keep last few turns only
    messages.append({"role": "user", "content": user_text})

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": LLM_MODEL, "messages": messages}

    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload)
    resp.raise_for_status()
    reply = resp.json()["choices"][0]["message"]["content"].strip()

    # update memory
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    conversation_memory[device_id] = history[-10:]

    return reply


def text_to_speech(text, lang=None):
    """Convert text to speech audio bytes using Edge-TTS (free, supports male voices).
    If lang not given, auto-detect based on script (Bangla vs English)."""
    if lang is None:
        lang = detect_lang_for_tts(text)

    # Male voices (Edge neural voices - free, no API key needed)
    voice = "bn-BD-PradeepNeural" if lang == "bn" else "en-US-GuyNeural"

    async def generate():
        buf = io.BytesIO()
        communicate = edge_tts.Communicate(text, voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        buf.seek(0)
        return buf

    return asyncio.run(generate())


# ---------------- ROUTES ----------------

@app.route("/", methods=["GET"])
def home():
    return "ARC Assistant Server is running."


@app.route("/process", methods=["POST"])
def process():
    """
    Expects: multipart/form-data with:
      - audio: the recorded WAV file
      - device_id (optional): to keep separate conversation memory
      - temp / humidity (optional): sensor readings from DHT11
    Returns: JSON with 'reply_text', plus an 'audio_url' style flow
             (for simplicity here we return MP3 audio directly).
    """
    if "audio" not in request.files:
        return jsonify({"error": "no audio file provided"}), 400

    audio_file = request.files["audio"]
    device_id = request.form.get("device_id", "default")
    temp = request.form.get("temp")
    humidity = request.form.get("humidity")

    sensor_context = None
    if temp or humidity:
        sensor_context = f"temperature={temp}C, humidity={humidity}%"

    try:
        user_text = transcribe_audio(audio_file.read(), audio_file.filename)
        if not user_text:
            return jsonify({"error": "could not understand audio"}), 400

        reply_text = get_llm_reply(user_text, device_id, sensor_context)
        audio_buf = text_to_speech(reply_text)

        # Send back audio directly; text is included in header for LCD display
        response = send_file(audio_buf, mimetype="audio/mpeg")
        response.headers["X-Reply-Text"] = reply_text.encode("utf-8").decode("latin-1", errors="ignore")
        response.headers["X-User-Text"] = user_text.encode("utf-8").decode("latin-1", errors="ignore")
        return response

    except requests.exceptions.HTTPError as e:
        return jsonify({"error": "API error", "detail": str(e)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/process_text_only", methods=["POST"])
def process_text_only():
    """
    Lightweight endpoint for testing without audio input/output.
    Send JSON: {"text": "...", "device_id": "...", "temp": "...", "humidity": "..."}
    Returns JSON reply text only (no TTS) - useful for quick debugging.
    """
    data = request.get_json(force=True)
    user_text = data.get("text", "")
    device_id = data.get("device_id", "default")
    sensor_context = None
    if data.get("temp") or data.get("humidity"):
        sensor_context = f"temperature={data.get('temp')}C, humidity={data.get('humidity')}%"

    if not user_text:
        return jsonify({"error": "no text provided"}), 400

    try:
        reply_text = get_llm_reply(user_text, device_id, sensor_context)
        return jsonify({"reply": reply_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
