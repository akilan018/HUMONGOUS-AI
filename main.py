# main.py
import os
import uuid
import httpx
from datetime import datetime
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware

import motor.motor_asyncio

# Import your existing NLP engine (must provide get_intent and get_response)
from nlp_engine import NlpEngine

# ----------------------------
# Configuration (from .env)
# ----------------------------
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-latest:generateContent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "MySuperSecretPassword123")

# ----------------------------
# App + Templates + Static
# ----------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")
# optional static dir
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
# Mongo (motor async client)
# ----------------------------
if not MONGO_URI:
    raise RuntimeError("MONGO_URI is not set in environment")

mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = mongo_client.get_default_database() if mongo_client is not None else mongo_client
# Fallback to explicit name if env URI doesn't set a DB: use 'humongous_ai'
if not db or db.name is None or db.name == "admin":
    db = mongo_client["humongous_ai"]
chat_collection = db["chat_logs"]

# Simple ping test at startup (will print in logs)
try:
    mongo_client.admin.command("ping")
    print("✅ MongoDB connected (ping OK).")
except Exception as e:
    print(f"❌ MongoDB ping failed: {e}")

# ----------------------------
# NLP Engine
# ----------------------------
try:
    engine = NlpEngine(intents_file="intents.json")
    print("✅ NLP Engine initialized.")
except Exception as e:
    print(f"❌ Failed to init NLP Engine: {e}")
    engine = None

# ----------------------------
# Utilities: DB helpers
# ----------------------------
async def log_interaction(session_id: str, sender: str, message: str, response: str = None, intent: str = None):
    """Insert one chat log document (async)."""
    doc = {
        "session_id": session_id,
        "timestamp": datetime.utcnow(),
        "sender": sender,
        "message": message,
    }
    if response is not None:
        doc["response"] = response
    if intent is not None:
        doc["intent"] = intent
    try:
        res = await chat_collection.insert_one(doc)
        # debug print — visible in Render logs / terminal
        print(f"✅ Logged chat (id={res.inserted_id})")
    except Exception as e:
        print(f"⚠️ Failed to log interaction: {e}")

async def get_all_chat_logs(limit: int = 1000) -> List[Dict[str, Any]]:
    """Return list of logs (most recent first)."""
    cursor = chat_collection.find().sort("timestamp", -1).limit(limit)
    docs = []
    async for d in cursor:
        d.pop("_id", None)
        # format timestamp for frontend
        ts = d.get("timestamp")
        if isinstance(ts, datetime):
            d["timestamp"] = ts.strftime("%Y-%m-%d %H:%M:%S")
        docs.append(d)
    return docs

# ----------------------------
# Gemini call (async)
# ----------------------------
async def call_gemini_api(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "**Error:** GEMINI_API_KEY not configured."
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", json=payload, headers=headers)
            resp.raise_for_status()
            result = resp.json()
            if "candidates" in result and result["candidates"]:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            return "I couldn't generate a response right now."
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return "AI service currently unavailable."

# ----------------------------
# Routes: index + admin
# ----------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, password: str):
    """Password protected admin page. Use ?password=..."""
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return templates.TemplateResponse("admin_dashboard.html", {"request": request})

@app.get("/api/admin/stats")
async def api_admin_stats():
    """Return stats used by admin dashboard (JSON)."""
    logs = await get_all_chat_logs(limit=2000)
    total_messages = len(logs)
    unique_sessions = len(set(log.get("session_id") for log in logs))
    fallback_count = sum(1 for l in logs if (l.get("intent") == "fallback" or l.get("intent") == "unknown"))
    fallback_rate = round((fallback_count / total_messages) * 100, 2) if total_messages else 0.0
    recent_logs = logs[:10]
    return JSONResponse({
        "total_messages": total_messages,
        "unique_sessions": unique_sessions,
        "fallback_rate": fallback_rate,
        "recent_logs": recent_logs
    })

# ----------------------------
# WebSocket chat
# ----------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    print(f"⚡ WebSocket opened session={session_id}")

    if not engine:
        await websocket.send_json({"type": "error", "message": "NLP engine not available."})
        await websocket.close()
        return

    # send greeting
    try:
        greeting = engine.get_response(engine.get_intent("hello"))
    except Exception:
        greeting = "Hello — I'm Humongous AI."
    await websocket.send_json({"type": "chat", "message": greeting})
    # Log greeting as bot message
    await log_interaction(session_id, "bot", greeting, response=None, intent="hello")

    try:
        while True:
            user_text = await websocket.receive_text()
            print(f"🗣 User ({session_id}): {user_text}")
            await log_interaction(session_id, "user", user_text)

            # determine intent & basic response
            matched_intent = engine.get_intent(user_text)
            intent_tag = matched_intent.get("tag", "unknown") if matched_intent else "unknown"

            COMPLEX_INTENTS = {
                'creator', 'creator_details', 'who_are_you', 'capabilities', 'company_info',
                'hours', 'location', 'origin', 'payments', 'returns', 'shipping', 'tracking',
                'order_management', 'discounts', 'technical_support', 'account_issues',
                'privacy_policy', 'product_info', 'feedback', 'human_handoff', 'billing_issues'
            }

            if intent_tag in COMPLEX_INTENTS:
                context_answer = engine.get_response(matched_intent)
                prompt = f"""
You are Humongous AI, a friendly assistant created by Akilan S R.
Answer the user's question using ONLY the context below. Do not invent facts.

CONTEXT:
"{context_answer}"

USER QUESTION:
"{user_text}"

ANSWER:
"""
                bot_text = await call_gemini_api(prompt)
                bot_text = f"✨ {bot_text}"
            else:
                bot_text = engine.get_response(matched_intent)

            # send and log
            await websocket.send_json({"type": "chat", "message": bot_text})
            await log_interaction(session_id, "bot", bot_text, response=bot_text, intent=intent_tag)
            print(f"✅ Responded (intent={intent_tag})")
    except WebSocketDisconnect:
        print(f"🔌 Session {session_id} disconnected.")
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass

# ----------------------------
# Start (if run directly)
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)