# main.py
import os
import uuid
import httpx
from datetime import datetime
from typing import List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware

from pymongo import MongoClient
from nlp_engine import NlpEngine  # your existing engine

# ----------------------------
# CONFIG
# ----------------------------
load_dotenv()
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-latest:generateContent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")  # default admin password

# ----------------------------
# FASTAPI SETUP
# ----------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# MONGO (Sync)
# ----------------------------
if not MONGO_URI:
    raise RuntimeError("❌ Missing MONGO_URI in environment")

try:
    mongo_client = MongoClient(MONGO_URI)
    mongo_client.admin.command("ping")
    print("✅ MongoDB connected successfully")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    mongo_client = None

db = None
if mongo_client:
    try:
        db = mongo_client.get_default_database()
    except Exception:
        db = None

if db is None or (hasattr(db, "name") and db.name == "admin"):
    db = mongo_client["humongous_ai"]

chat_collection = db["chat_logs"]

# ----------------------------
# NLP ENGINE
# ----------------------------
try:
    engine = NlpEngine(intents_file="intents.json")
    print("✅ NLP Engine initialized")
except Exception as e:
    print(f"❌ NLP Engine init failed: {e}")
    engine = None

# ----------------------------
# HELPERS
# ----------------------------
def _format_timestamp(ts) -> str:
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return str(ts)

def log_interaction(session_id: str, sender: str, message: str, response: str = None, intent: str = None):
    doc: Dict[str, Any] = {
        "session_id": session_id,
        "timestamp": datetime.utcnow(),
        "sender": sender,
        "message": message,
    }
    if response:
        doc["response"] = response
    if intent:
        doc["intent"] = intent
    try:
        chat_collection.insert_one(doc)
    except Exception as e:
        print(f"⚠️ Failed to log chat: {e}")

def get_all_chat_logs(limit: int = 0) -> List[Dict[str, Any]]:
    try:
        cursor = chat_collection.find().sort("timestamp", -1)
        if limit > 0:
            cursor = cursor.limit(limit)
        logs = []
        for d in cursor:
            d.pop("_id", None)
            d["timestamp"] = _format_timestamp(d.get("timestamp"))
            logs.append(d)
        return logs
    except Exception as e:
        print(f"⚠️ Failed to get logs: {e}")
        return []

async def call_gemini_api(prompt: str) -> str:
    """Call Gemini strictly for company FAQs only (not general questions)."""
    if not GEMINI_API_KEY:
        return "⚠️ Missing GEMINI_API_KEY"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", json=payload, headers=headers)
            res.raise_for_status()
            data = res.json()
            if isinstance(data, dict) and "candidates" in data and data["candidates"]:
                try:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                except Exception:
                    return "I'm not sure about that. Please ask something related to our services."
            return "I'm not sure about that. Please ask something related to our services."
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return "AI service temporarily unavailable."

# ----------------------------
# ROUTES
# ----------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, password: str):
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return templates.TemplateResponse("admin_dashboard.html", {"request": request})

@app.get("/api/admin/stats")
def admin_stats():
    try:
        logs = get_all_chat_logs()
        total = len(logs)
        unique_sessions = len({l.get("session_id") for l in logs if l.get("session_id")})
        fallback_count = sum(1 for l in logs if l.get("intent") in ["fallback", "unknown"])
        fallback_rate = round((fallback_count / total) * 100, 2) if total else 0.0

        msg_timeline: Dict[str, int] = {}
        for l in logs:
            date_key = l.get("timestamp", "")[:10]
            if date_key:
                msg_timeline[date_key] = msg_timeline.get(date_key, 0) + 1

        recent_logs = []
        for l in logs:
            recent_logs.append({
                "timestamp": l.get("timestamp", ""),
                "session_id": l.get("session_id"),
                "sender": l.get("sender"),
                "message": l.get("message"),
                "response": l.get("response", ""),
                "intent": l.get("intent", "")
            })

        return JSONResponse({
            "total_messages": total,
            "unique_sessions": unique_sessions,
            "fallback_rate": fallback_rate,
            "recent_logs": recent_logs,
            "timeline": msg_timeline
        })
    except Exception as e:
        print(f"⚠️ Dashboard data error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# ----------------------------
# WEBSOCKET CHAT
# ----------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    print(f"⚡ WebSocket connected: {session_id}")

    greeting = "Hello! I'm Humongous AI, your assistant."
    if engine:
        try:
            greeting = engine.get_response(engine.get_intent("hello"))
        except Exception:
            pass

    await websocket.send_json({"type": "chat", "message": greeting})
    log_interaction(session_id, "bot", greeting, intent="hello")

    try:
        while True:
            user_msg = await websocket.receive_text()
            print(f"🗣 User: {user_msg}")
            log_interaction(session_id, "user", user_msg)

            intent_tag = "unknown"
            matched = None
            if engine:
                try:
                    matched = engine.get_intent(user_msg)
                    intent_tag = matched.get("tag", "unknown") if isinstance(matched, dict) else str(matched)
                except Exception:
                    intent_tag = "unknown"

            # Prefer local intent responses (company FAQs)
            if engine and intent_tag not in ["fallback", "unknown"]:
                try:
                    bot_msg = engine.get_response(matched)
                except Exception:
                    bot_msg = "Sorry, I couldn’t find that answer right now."
            else:
                # Strictly restrict Gemini to company-related answers
                company_context = (
                    "You are Humongous AI, an assistant for Akilan S R's company. "
                    "Only answer customer-related FAQs such as pricing, services, support, and company policies. "
                    "If the user asks unrelated or general questions, politely say: "
                    "'I'm only trained to answer questions about our services and support.'"
                )
                prompt = f"{company_context}\nUser: {user_msg}\nAnswer clearly and briefly."
                bot_msg = await call_gemini_api(prompt)

            await websocket.send_json({"type": "chat", "message": bot_msg})
            log_interaction(session_id, "bot", bot_msg, response=bot_msg, intent=intent_tag)

    except WebSocketDisconnect:
        print(f"🔌 Disconnected: {session_id}")
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass

# ----------------------------
# START
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)