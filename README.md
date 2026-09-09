# ARC Assistant - Cloud Server

ESP8266 এর জন্য cloud-based AI assistant backend।
Flow: Audio (ESP8266) → Groq Whisper (STT) → OpenRouter LLM → gTTS (TTS) → Audio back to ESP8266

## 1. API Keys সংগ্রহ (দুটোই ফ্রি)

- **Groq**: https://console.groq.com/keys → ফ্রি sign up → API key কপি করুন
- **OpenRouter**: https://openrouter.ai/keys → ফ্রি sign up → API key কপি করুন
  - Free models list: https://openrouter.ai/models?max_price=0

## 2. Local এ টেস্ট করা (optional, deploy করার আগে)

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_groq_key"
export OPENROUTER_API_KEY="your_openrouter_key"
python app.py
```

তারপর test করুন:
```bash
curl -X POST http://localhost:5000/process_text_only \
  -H "Content-Type: application/json" \
  -d '{"text": "আজকে ঘরের তাপমাত্রা কেমন?", "temp": "29", "humidity": "60"}'
```

## 3. GitHub এ Push করা

```bash
git init
git add .
git commit -m "ARC assistant server"
git branch -M main
git remote add origin https://github.com/<your-username>/arc-server.git
git push -u origin main
```

## 4. Render.com এ Deploy করা (ফ্রি)

1. https://render.com এ sign up করুন (GitHub দিয়ে login করলে সহজ হবে)
2. Dashboard → "New +" → "Web Service"
3. আপনার GitHub repo সিলেক্ট করুন
4. Settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
   - **Instance Type**: Free
5. Environment Variables যোগ করুন (Environment tab এ):
   - `GROQ_API_KEY` = আপনার groq key
   - `OPENROUTER_API_KEY` = আপনার openrouter key
6. "Create Web Service" ক্লিক করুন

Deploy শেষ হলে আপনি একটা URL পাবেন যেমন:
`https://arc-server-xxxx.onrender.com`

এটাই ESP8266 কোডে বসাতে হবে।

## 5. Server কে সবসময় Active রাখা (Free tier sleep prevent)

Render free tier ১৫ মিনিট idle থাকলে sleep করে। এটা ঠেকাতে:

1. https://uptimerobot.com এ ফ্রি account বানান
2. New Monitor → HTTP(s) → আপনার Render URL দিন
3. প্রতি ৫ মিনিটে ping করবে, server ঘুমাবে না

## 6. Endpoints

### `POST /process`
Audio পাঠান, audio + text reply ফেরত পাবেন।
- Form fields: `audio` (WAV file), `device_id` (optional), `temp`, `humidity` (optional)
- Response: MP3 audio, headers এ `X-Reply-Text` ও `X-User-Text` থাকবে

### `POST /process_text_only`
টেস্টিং এর জন্য, শুধু text দিয়ে LLM রিপ্লাই পাবেন (no audio)।
- JSON body: `{"text": "...", "device_id": "...", "temp": "...", "humidity": "..."}`

## Notes

- **ভাষা: Bangla + English দুটোই সাপোর্ট করে (mixed/Banglish)।**
  - STT (Groq Whisper): কোনো fixed language দেওয়া নেই, তাই এটা automatic বুঝে নেয় Bangla/English/মিশ্রিত স্পিচ।
  - LLM কে বলা আছে ইউজার যেভাবে বলবে (বাংলা/ইংরেজি/মিক্স) সেভাবেই রিপ্লাই দিতে।
  - TTS (gTTS): reply এর মধ্যে বাংলা অক্ষর বেশি থাকলে Bangla ভয়েস, না হলে English ভয়েস ব্যবহার হয় (auto-detect, `detect_lang_for_tts` ফাংশনে)।
  - Note: gTTS সত্যিকারের code-mixed (এক বাক্যে বাংলা+ইংরেজি) audio ভালোভাবে বলতে পারে না, তাই LLM কে prompt এ বলা আছে প্রতিটা reply মূলত একটা ভাষায় রাখতে যাতে TTS পরিষ্কার শোনায়।
- `LLM_MODEL` ভ্যারিয়েবল বদলে অন্য ফ্রি OpenRouter model ব্যবহার করতে পারেন।
- Conversation memory এখন শুধু RAM এ থাকে (server restart হলে মুছে যাবে) — এটা basic version, পরে database দিয়ে persistent করা যাবে।
