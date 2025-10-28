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
from nlp_engine import NlpEngine

# ----------------------------
# CONFIG
# ----------------------------
load_dotenv()
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-latest:generateContent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

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
    raise RuntimeError("❌ Missing MONGO_URI in environment!")

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
    print(f"❌ NLP Engine init failed: {e}")
    engine = None

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def log_interaction(session_id, sender, message, response=None, intent=None):
    """Store chat logs in MongoDB"""
    doc = {
        "session_id": session_id,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "sender": sender,
        "message": message,
    }
    if response:
        doc["response"] = response
    if intent:
        doc["intent"] = intent
    try:
        chat_collection.insert_one(doc)
        print(f"✅ Logged chat ({sender})")
    except Exception as e:
        print(f"⚠️ Failed to log chat: {e}")

def get_all_chat_logs(limit=1000):
    try:
        docs = chat_collection.find().sort("timestamp", -1).limit(limit)
        logs = []
        for d in docs:
            d.pop("_id", None)
            logs.append(d)
        return logs
    except Exception as e:
        print(f"⚠️ Failed to get logs: {e}")
        return []

async def call_gemini_api(prompt: str):
    if not GEMINI_API_KEY:
        return "⚠️ Missing GEMINI_API_KEY"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", json=payload, headers=headers)
            res.raise_for_status()
            data = res.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
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
        logs = get_all_chat_logs(limit=2000)
        total = len(logs)
        unique_sessions = len(set(l.get("session_id") for l in logs if l.get("session_id")))
        fallback_count = sum(1 for l in logs if l.get("intent") in ["fallback", "unknown"])
        fallback_rate = round((fallback_count / total) * 100, 2) if total else 0.0

        # Message timeline (by date)
        msg_timeline = {}
        for l in logs:
            ts = l.get("timestamp", "")[:10]
            if ts:
                msg_timeline[ts] = msg_timeline.get(ts, 0) + 1

        return JSONResponse({
            "total_messages": total,
            "unique_sessions": unique_sessions,
            "fallback_rate": fallback_rate,
            "recent_logs": logs[:10],
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

    greeting = "Hello! I'm Humongous AI."
    await websocket.send_json({"type": "chat", "message": greeting})
    log_interaction(session_id, "bot", greeting, intent="hello")

    try:
        while True:
            user_msg = await websocket.receive_text()
            log_interaction(session_id, "user", user_msg)

            matched = engine.get_intent(user_msg)
            intent_tag = matched.get("tag", "unknown") if matched else "unknown"

            if intent_tag not in ["fallback", "unknown"]:
                bot_msg = engine.get_response(matched)
            else:
                prompt = f"User: {user_msg}\nYou are Humongous AI created by Akilan S R. Respond helpfully."
                bot_msg = await call_gemini_api(prompt)

            await websocket.send_json({"type": "chat", "message": bot_msg})
            log_interaction(session_id, "bot", bot_msg, intent=intent_tag)

    except WebSocketDisconnect:
        print(f"🔌 Disconnected: {session_id}")
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")
        await websocket.close()

# ----------------------------
# START SERVER
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)