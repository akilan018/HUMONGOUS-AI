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
from nlp_engine import NlpEngine  # your NLP engine class

# ----------------------------
# CONFIG
# ----------------------------
load_dotenv()
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-latest:generateContent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

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
# MONGO SETUP
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

db = mongo_client.get_default_database() if mongo_client else None
if db is None or db.name == "admin":
    db = mongo_client["humongous_ai"]

chat_collection = db["chat_logs"]

# ----------------------------
# NLP ENGINE
# ----------------------------
try:
    engine = NlpEngine(intents_file="intents.json")
    print("✅ NLP Engine initialized")
except Exception as e:
    print(f"⚠️ NLP Engine init failed: {e}")
    engine = None

# ----------------------------
# HELPERS
# ----------------------------
def _format_timestamp(ts) -> str:
    """Ensure timestamps are always stringified."""
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(ts, str):
        return ts
    return str(ts)

def log_interaction(session_id: str, sender: str, message: str, response: str = None, intent: str = None):
    """Save chat logs into MongoDB"""
    doc = {
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
        print(f"⚠️ Log insert failed: {e}")

def get_all_chat_logs(limit: int = 0) -> List[Dict[str, Any]]:
    """Fetch logs safely"""
    try:
        cursor = chat_collection.find().sort("timestamp", -1)
        if limit:
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
    """Call Gemini API safely"""
    if not GEMINI_API_KEY:
        return "⚠️ Missing Gemini API Key"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", json=payload, headers=headers)
            res.raise_for_status()
            data = res.json()
            return (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "I'm only trained to answer questions about our services.")
            )
    except Exception as e:
        print(f"⚠️ Gemini API error: {e}")
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

    logs = get_all_chat_logs(limit=20)
    analytics = {
        "total_messages": len(logs),
        "unique_sessions": len({l.get("session_id") for l in logs}),
        "fallback_rate": round(
            (sum(1 for l in logs if l.get("intent") in ["fallback", "unknown"]) / len(logs)) * 100, 2
        )
        if logs else 0.0,
        "recent_logs": logs,
    }
    return templates.TemplateResponse("admin_dashboard.html", {"request": request, "analytics": analytics})

@app.get("/api/admin/stats")
def admin_stats():
    try:
        logs = get_all_chat_logs()
        total = len(logs)
        unique_sessions = len({l.get("session_id") for l in logs if l.get("session_id")})
        fallback_count = sum(1 for l in logs if l.get("intent") in ["fallback", "unknown"])
        fallback_rate = round((fallback_count / total) * 100, 2) if total else 0.0

        msg_timeline = {}
        for l in logs:
            date_key = l.get("timestamp", "")[:10]
            if date_key:
                msg_timeline[date_key] = msg_timeline.get(date_key, 0) + 1

        return JSONResponse({
            "total_messages": total,
            "unique_sessions": unique_sessions,
            "fallback_rate": fallback_rate,
            "recent_logs": logs,
            "timeline": msg_timeline
        })
    except Exception as e:
        print(f"⚠️ Dashboard data error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# ----------------------------
# WEBSOCKET
# ----------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    print(f"⚡ Connected session: {session_id}")

    greeting = "Hello! I'm Humongous AI, your assistant."
    await websocket.send_json({"type": "chat", "message": greeting})
    log_interaction(session_id, "bot", greeting, intent="hello")

    try:
        while True:
            user_msg = await websocket.receive_text()
            log_interaction(session_id, "user", user_msg)

            intent_tag = "unknown"
            if engine:
                try:
                    matched = engine.get_intent(user_msg)
                    intent_tag = matched.get("tag", "unknown")
                    bot_msg = engine.get_response(matched)
                except Exception:
                    intent_tag = "unknown"
                    bot_msg = None
            else:
                bot_msg = None

            if not bot_msg or intent_tag in ["fallback", "unknown"]:
                company_context = (
                    "You are Humongous AI, an assistant for Akilan S R's company. "
                    "Only answer customer-related FAQs like pricing, services, or support. "
                    "If unrelated, say 'I'm only trained to answer questions about our services.'"
                )
                prompt = f"{company_context}\nUser: {user_msg}"
                bot_msg = await call_gemini_api(prompt)

            await websocket.send_json({"type": "chat", "message": bot_msg})
            log_interaction(session_id, "bot", bot_msg, response=bot_msg, intent=intent_tag)

    except WebSocketDisconnect:
        print(f"🔌 Disconnected: {session_id}")

# ----------------------------
# ERROR HANDLER
# ----------------------------
@app.exception_handler(Exception)
async def all_exception_handler(request, exc):
    import traceback
    print("🔥 ERROR:", traceback.format_exc())
    return JSONResponse({"error": str(exc)}, status_code=500)

# ----------------------------
# START
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)