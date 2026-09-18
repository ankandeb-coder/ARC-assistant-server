import os
import io
import json
import re
import asyncio
import requests
from flask import Flask, request, send_file, jsonify
import edge_tts

app = Flask(__name__)

# ---------------- CONFIG ----------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
GOOGLE_DRIVE_API_KEY = os.environ.get("GOOGLE_DRIVE_API_KEY", "")      # optional - your own songs on Drive
GOOGLE_DRIVE_FOLDER_ID = os.environ.get("GOOGLE_DRIVE_FOLDER_ID", "")  # the shared folder holding your mp3s

GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Fixed, pinned model choice: OpenAI's own open-weight model, purpose-trained
# for function calling / tool use (Harmony format). Hosted directly by OpenAI
# on OpenRouter's free tier, so it avoids most of the "no provider satisfies
# policy + tool-calling" 404 issues seen with other pinned third-party models.
LLM_MODEL = "openai/gpt-oss-20b:free"

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
    """Free web/news search via DuckDuckGo (no API key or signup needed).
    NOTE: This uses an unofficial scraping-based library (ddgs), since
    DuckDuckGo has no official free search API. It can occasionally get
    rate-limited or blocked (especially from cloud/datacenter IPs like
    Render's), in which case it will return an error message instead of
    crashing - the LLM will just tell the user search isn't available
    right now."""
    from ddgs import DDGS

    try:
        results = list(DDGS().news(query, max_results=3))
        if not results:
            # Fall back to general text search if no news results
            results = list(DDGS().text(query, max_results=3))
    except Exception as e:
        return f"Web search is temporarily unavailable ({e})."

    if not results:
        return f"No search results found for '{query}'."

    summary_parts = []
    for r in results[:3]:
        title = r.get("title", "")
        body = (r.get("body") or "")[:200]
        date = r.get("date", "")
        summary_parts.append(f"{title} ({date}): {body}" if date else f"{title}: {body}")

    return " | ".join(summary_parts)


def tool_find_song(mood_or_query):
    """Find a song to play. Order of preference:
    1. Your own Google Drive folder (your legally-owned mp3s - full songs, no copyright issue)
    2. Internet Archive's music collection (free, legal, but limited/older catalog)
    """
    if GOOGLE_DRIVE_API_KEY and GOOGLE_DRIVE_FOLDER_ID:
        audio_url, description = search_google_drive(mood_or_query)
        if audio_url:
            return audio_url, description

    return search_internet_archive(mood_or_query)


def search_google_drive(query):
    """Search your shared Google Drive folder for a matching mp3 by filename.
    Folder must be shared as 'Anyone with the link - Viewer'."""
    list_resp = requests.get(
        "https://www.googleapis.com/drive/v3/files",
        params={
            "q": f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents and name contains '{query}' and trashed = false",
            "key": GOOGLE_DRIVE_API_KEY,
            "fields": "files(id, name)",
        }
    )
    list_resp.raise_for_status()
    files = list_resp.json().get("files", [])

    if not files:
        # Try a looser search: just list any mp3 in the folder if no name match
        list_resp = requests.get(
            "https://www.googleapis.com/drive/v3/files",
            params={
                "q": f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents and trashed = false",
                "key": GOOGLE_DRIVE_API_KEY,
                "fields": "files(id, name)",
                "pageSize": 50,
            }
        )
        list_resp.raise_for_status()
        files = list_resp.json().get("files", [])

    if not files:
        return None, None

    chosen = files[0]
    file_id = chosen["id"]
    title = chosen.get("name", "Unknown")

    audio_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={GOOGLE_DRIVE_API_KEY}"
    return audio_url, f"{title} (from your Google Drive)"


def search_internet_archive(mood_or_query):
    """Free FULL-length song search via the Internet Archive (archive.org).
    No signup or API key needed. Fallback source if Google Drive has no match."""

    def search_archive(query):
        search_resp = requests.get(
            "https://archive.org/advancedsearch.php",
            params={
                "q": f'({query}) AND mediatype:(audio) AND collection:(audio_music)',
                "fl[]": "identifier",
                "rows": 1,
                "sort[]": "downloads desc",
                "output": "json",
            }
        )
        search_resp.raise_for_status()
        return search_resp.json().get("response", {}).get("docs", [])

    docs = search_archive(mood_or_query)
    if not docs:
        docs = search_archive("song")
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


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Dhaka' or 'Kolkata'"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information, news, or facts not known in advance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query, in English for best results"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_song",
            "description": "Find and play a full song or piece of music.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Mood, genre, or song description, in English keywords"}
                },
                "required": ["query"],
            },
        },
    },
]


def run_tool(name, args):
    """Execute a tool by name and return (result_text, song_audio_url_or_None)."""
    if name == "get_weather":
        return tool_get_weather(args.get("city", "")), None
    elif name == "web_search":
        return tool_web_search(args.get("query", "")), None
    elif name == "play_song":
        audio_url, result_text = tool_find_song(args.get("query", ""))
        return result_text, audio_url
    else:
        return f"Unknown tool: {name}", None


# ---------------- DEFENSIVE FALLBACK: catch leaked raw tool-call text ----------------
# Some weaker free models don't properly use OpenRouter's structured tool_calls field
# and instead leak their own internal tool-call syntax as plain text content, e.g.:
#   <|tool_call_start|>[[search(query='...')]<|tool_call_end|>
#   <tool_call>web_search\n<arg_key>query</arg_key>\n<arg_value>...</arg_value>\n</tool_call>
# These patterns catch the common leaked formats and extract (tool_name, argument)
# so we can still run the right tool instead of showing garbage text to the user.
_LEAK_PATTERNS = [
    re.compile(r"(\w+)\s*\(\s*(?:query|city|argument)\s*=\s*['\"](.+?)['\"]\s*\)", re.IGNORECASE),
    re.compile(r"<tool_call>\s*(\w+).*?<arg_value>(.+?)</arg_value>", re.IGNORECASE | re.DOTALL),
]
_LEAK_NAME_MAP = {
    "search": "web_search", "web_search": "web_search",
    "weather": "get_weather", "get_weather": "get_weather",
    "song": "play_song", "play_song": "play_song",
}


def detect_leaked_tool_call(content):
    """Returns (tool_name, argument) if leaked tool-call text is found, else None."""
    if not content:
        return None
    for pattern in _LEAK_PATTERNS:
        match = pattern.search(content)
        if match:
            raw_name, arg = match.group(1), match.group(2)
            mapped_name = _LEAK_NAME_MAP.get(raw_name.lower())
            if mapped_name:
                return mapped_name, arg.strip()
    return None


def get_llm_reply(user_text, device_id="default", sensor_context=None):
    """Send text to OpenRouter LLM, run a tool via OpenRouter's native function-calling
    API if requested, and return the final reply.
    Returns a tuple: (reply_text, song_audio_url_or_None)."""
    history = conversation_memory.get(device_id, [])

    system_prompt = (
        "You are ARC, a helpful voice assistant running on an ESP8266 smart device. "
        "Keep replies short (1-3 sentences), clear, and conversational, "
        "since they will be spoken aloud and shown on a small LCD screen. "
        "The user may speak in Bangla, English, or mixed Banglish - reply naturally "
        "in whichever language(s) the user used, matching their style. "
        "Prefer replying mostly in one dominant language (Bangla OR English) per response "
        "so the reply can be converted to speech cleanly, but you may mix a few words if natural. "
        "Only call a tool when the user's request genuinely needs it (current weather, "
        "recent news/events, or wanting to hear music) - otherwise just answer directly."
    )
    if sensor_context:
        system_prompt += f" Current sensor readings: {sensor_context}."

    vision_desc = latest_vision.get(device_id)
    if vision_desc:
        system_prompt += (
            f" A camera near you last saw: \"{vision_desc}\". "
            "Only mention this if the user's question is actually about what you can see."
        )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_text})

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    def call_llm(msgs, use_tools=True):
        payload = {"model": LLM_MODEL, "messages": msgs, "max_tokens": 300}
        if use_tools:
            payload["tools"] = TOOL_DEFINITIONS
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices")
        if not choices:
            raise RuntimeError(f"OpenRouter returned no choices: {data}")
        return choices[0].get("message", {})

    assistant_message = call_llm(messages)
    song_audio_url = None
    tool_calls = assistant_message.get("tool_calls")
    leaked = None if tool_calls else detect_leaked_tool_call(assistant_message.get("content"))

    if tool_calls:
        # Only handle the first tool call for simplicity (our tools are single-step)
        call = tool_calls[0]
        fn_name = call["function"]["name"]
        try:
            fn_args = json.loads(call["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            fn_args = {}

        tool_result_text, song_audio_url = run_tool(fn_name, fn_args)

        # Reply to the model with the tool's result, using the proper "tool" role,
        # so it can produce a natural final answer.
        followup_messages = messages + [
            assistant_message,
            {
                "role": "tool",
                "tool_call_id": call.get("id", "call_1"),
                "content": tool_result_text,
            },
        ]
        final_message = call_llm(followup_messages, use_tools=False)
        final_reply = final_message.get("content") or tool_result_text

    elif leaked:
        # The model leaked its own raw tool-call syntax instead of using the
        # proper structured format - run the tool anyway based on what we parsed.
        fn_name, fn_arg = leaked
        arg_key = "city" if fn_name == "get_weather" else "query"
        tool_result_text, song_audio_url = run_tool(fn_name, {arg_key: fn_arg})

        followup_messages = messages + [
            {"role": "user", "content": f"[Tool result: {tool_result_text}] Answer the user naturally in 1-2 short sentences using this information."}
        ]
        final_message = call_llm(followup_messages, use_tools=False)
        final_reply = final_message.get("content") or tool_result_text

    else:
        final_reply = assistant_message.get("content") or "Sorry, I couldn't come up with a reply just now."

    final_reply = final_reply.strip()

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
    return "ARC Assistant Server is running. Code version: v2-tool-null-safety-fix"


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
        ],
        "max_tokens": 150,
    }

    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices")
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {data}")
    content = choices[0].get("message", {}).get("content")
    return (content or "Unable to describe the image right now.").strip()


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
