import os
import io
import re
import asyncio
import requests
from flask import Flask, request, send_file, jsonify
import edge_tts

app = Flask(__name__)

# ---------------- CONFIG ----------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")  # optional - web search tool won't work without this

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

# Latest sensor readings and camera description, shared across devices with same device_id
# (populated by ESP32-CAM's /vision uploads, read by ESP8266's /process voice queries)
latest_sensor = {}   # device_id -> {"temp": ..., "humidity": ...}
latest_vision = {}   # device_id -> description text

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


# ---------------- TOOLS (weather, web search, song) ----------------
# These are called when the LLM emits a special marker in its reply, e.g.
#   [TOOL: weather | Dhaka]
#   [TOOL: search | latest bangladesh cricket score]
#   [TOOL: song | relaxing lofi]
# The server detects the marker, runs the tool, then asks the LLM once more
# with the tool result so it can produce a natural final answer.

def tool_get_weather(location):
    """Free weather lookup via Open-Meteo (no API key needed).
    First geocodes the location name, then fetches current weather."""
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1}
    ).json()

    results = geo.get("results")
    if not results:
        return f"Could not find location '{location}'."

    lat = results[0]["latitude"]
    lon = results[0]["longitude"]
    place_name = results[0].get("name", location)

    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "current_weather": "true"}
    ).json()

    cw = weather.get("current_weather", {})
    if not cw:
        return f"Could not fetch weather for {place_name}."

    return (
        f"Weather in {place_name}: {cw.get('temperature')}°C, "
        f"wind {cw.get('windspeed')} km/h."
    )


def tool_web_search(query):
    """Free-tier web search via Tavily. Returns a short text summary of top results."""
    if not TAVILY_API_KEY:
        return "Web search is not configured on this server (missing TAVILY_API_KEY)."

    resp = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 3, "search_depth": "basic"}
    )
    resp.raise_for_status()
    data = resp.json()

    results = data.get("results", [])
    if not results:
        return f"No search results found for '{query}'."

    summary_parts = []
    for r in results[:3]:
        title = r.get("title", "")
        content = r.get("content", "")[:200]
        summary_parts.append(f"{title}: {content}")

    return " | ".join(summary_parts)


def tool_find_song(mood_or_query):
    """Free FULL-length song search via the Internet Archive (archive.org).
    No signup or API key needed. Returns public-domain / openly licensed
    full tracks, not short previews."""

    def search_archive(query):
        search_resp = requests.get(
            "https://archive.org/advancedsearch.php",
            params={
                "q": f'({query}) AND mediatype:(audio)',
                "fl[]": "identifier",
                "rows": 1,
                "sort[]": "downloads desc",  # prefer popular/well-seeded items
                "output": "json",
            }
        )
        search_resp.raise_for_status()
        return search_resp.json().get("response", {}).get("docs", [])

    docs = search_archive(mood_or_query)
    if not docs:
        # Fallback to a generic, reliably-populated query if the specific one found nothing
        docs = search_archive("music")
    if not docs:
        return None, f"No song found for '{mood_or_query}'."

    identifier = docs[0]["identifier"]

    meta_resp = requests.get(f"https://archive.org/metadata/{identifier}")
    meta_resp.raise_for_status()
    meta = meta_resp.json()

    files = meta.get("files", [])
    mp3_file = next((f for f in files if f.get("name", "").lower().endswith(".mp3")), None)
    if not mp3_file:
        return None, f"Found '{identifier}' but no playable mp3 file."

    audio_url = f"https://archive.org/download/{identifier}/{mp3_file['name']}"
    title = meta.get("metadata", {}).get("title", identifier)

    return audio_url, f"{title} (full track, Internet Archive)"


TOOL_MARKER_RE = re.compile(r"\[TOOL:\s*(\w+)\s*\|\s*(.+?)\]")

TOOLS_SYSTEM_PROMPT = (
    "\n\nYou have access to these tools. To use one, reply with ONLY a line in this "
    "exact format and nothing else: [TOOL: name | argument]\n"
    "- [TOOL: weather | <city name>] - get current weather for a city\n"
    "- [TOOL: search | <search query>] - search the web for current information\n"
    "- [TOOL: song | <mood, genre, or song description, in English keywords>] - find and play a full song\n"
    "Only use a tool when the user's request actually needs it (e.g. asking about "
    "current weather, recent news/events, or wanting to hear music). "
    "Otherwise, just answer normally in plain conversational text."
)


def get_llm_reply(user_text, device_id="default", sensor_context=None):
    """Send text to OpenRouter LLM, run a tool if requested, and return the final reply.
    Returns a tuple: (reply_text, song_audio_url_or_None)."""
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

    vision_desc = latest_vision.get(device_id)
    if vision_desc:
        system_prompt += (
            f" A camera near you last saw: \"{vision_desc}\". "
            "Only mention this if the user's question is actually about what you can see."
        )

    system_prompt += TOOLS_SYSTEM_PROMPT

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_text})

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    def call_llm(msgs):
        payload = {"model": LLM_MODEL, "messages": msgs}
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    first_reply = call_llm(messages)
    song_audio_url = None

    tool_match = TOOL_MARKER_RE.search(first_reply)
    if tool_match:
        tool_name = tool_match.group(1).lower()
        tool_arg = tool_match.group(2).strip()

        if tool_name == "weather":
            tool_result = tool_get_weather(tool_arg)
        elif tool_name == "search":
            tool_result = tool_web_search(tool_arg)
        elif tool_name == "song":
            song_audio_url, tool_result = tool_find_song(tool_arg)
        else:
            tool_result = "Unknown tool requested."

        # Ask the LLM again, now with the tool's result, to produce a natural final reply
        followup_messages = messages + [
            {"role": "assistant", "content": first_reply},
            {"role": "user", "content": f"[TOOL RESULT: {tool_result}] Now answer the user naturally using this information, in 1-2 short sentences."}
        ]
        final_reply = call_llm(followup_messages)
    else:
        final_reply = first_reply

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": final_reply})
    conversation_memory[device_id] = history[-10:]

    return final_reply, song_audio_url


def text_to_speech(text, lang=None):
    """Convert text to speech audio bytes using Edge-TTS (free, supports male voices).
    If lang not given, auto-detect based on script (Bangla vs English)."""
    if lang is None:
        lang = detect_lang_for_tts(text)

    # Male voices (Edge neural voices - free, no API key needed)
    # bn-IN-BashkarNeural = Indian Bengali (West Bengal) male accent
    # en-US-GuyNeural = US English male
    voice = "bn-IN-BashkarNeural" if lang == "bn" else "en-US-GuyNeural"

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

    # Fall back to the latest sensor reading reported by the ESP32-CAM, if
    # this ESP8266 request didn't include its own temp/humidity values.
    if not temp and not humidity and device_id in latest_sensor:
        temp = latest_sensor[device_id].get("temp")
        humidity = latest_sensor[device_id].get("humidity")

    sensor_context = None
    if temp or humidity:
        sensor_context = f"temperature={temp}C, humidity={humidity}%"

    try:
        user_text = transcribe_audio(audio_file.read(), audio_file.filename)
        if not user_text:
            return jsonify({"error": "could not understand audio"}), 400

        reply_text, song_audio_url = get_llm_reply(user_text, device_id, sensor_context)

        if song_audio_url:
            # Proxy the song's mp3 directly instead of speaking it via TTS
            song_resp = requests.get(song_audio_url, stream=True)
            song_resp.raise_for_status()
            response = send_file(io.BytesIO(song_resp.content), mimetype="audio/mpeg")
            response.headers["X-Reply-Text"] = reply_text.encode("utf-8").decode("latin-1", errors="ignore")
            response.headers["X-User-Text"] = user_text.encode("utf-8").decode("latin-1", errors="ignore")
            response.headers["X-Is-Song"] = "true"
            return response

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


@app.route("/vision", methods=["POST"])
def vision():
    """
    Called by the ESP32-CAM. Expects multipart/form-data with:
      - image: JPEG file from the camera
      - device_id (optional): to link this camera to the same "assistant" as an ESP8266
      - temp / humidity (optional): DHT11 readings, now living on the ESP32-CAM board

    Stores the camera description + sensor readings in memory (keyed by device_id),
    so the ESP8266's /process voice endpoint can reference them later
    (e.g. "what do you see?" or "what's the temperature?").

    Returns a short JSON description - the ESP32-CAM can ignore this or use it
    to flash its RGB LED, etc.
    """
    if "image" not in request.files:
        return jsonify({"error": "no image file provided"}), 400

    image_file = request.files["image"]
    device_id = request.form.get("device_id", "default")
    temp = request.form.get("temp")
    humidity = request.form.get("humidity")

    if temp or humidity:
        latest_sensor[device_id] = {"temp": temp, "humidity": humidity}

    try:
        image_bytes = image_file.read()
        description = describe_image(image_bytes)
        latest_vision[device_id] = description
        return jsonify({"description": description})
    except requests.exceptions.HTTPError as e:
        return jsonify({"error": "API error", "detail": str(e)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def describe_image(image_bytes):
    """Send an image to a free vision-capable OpenRouter model and get a short description."""
    import base64
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "openrouter/free",  # auto-router picks a free model that supports images
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe what is in this image in one short sentence, "
                                "as if reporting what a home camera just saw."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                    }
                ]
            }
        ]
    }

    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


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
        reply_text, song_audio_url = get_llm_reply(user_text, device_id, sensor_context)
        result = {"reply": reply_text}
        if song_audio_url:
            result["song_audio_url"] = song_audio_url
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
