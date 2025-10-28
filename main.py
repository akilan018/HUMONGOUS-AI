
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
from nlp_engine import NlpEngine

# ----------------------------
# Configuration
# ----------------------------
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-latest:generateContent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# ----------------------------
# App setup
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
# MongoDB (Motor)
# ----------------------------
if not MONGO_URI:
    raise RuntimeError("MONGO_URI is missing in environment")

mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
try:
    mongo_client.admin.command("ping")
    print("✅ MongoDB connected (ping OK).")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")

# Ensure correct database selection (Render/Mongo Atlas safe)
db = mongo_client.get_default_database()
if not db or db.name is None or db.name == "admin":
    db = mongo_client["humongous_ai"]

chat_collection = db["chat_logs"]

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
# Utility: Logging & Fetching
# ----------------------------
async def log_interaction(session_id, sender, message, response=None, intent=None):
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
        await chat_collection.insert_one(doc)
        print(f"✅ Logged chat for {sender}")
    except Exception as e:
        print(f"⚠️ Log failed: {e}")

async def get_all_chat_logs(limit=1000):
    cursor = chat_collection.find().sort("timestamp", -1).limit(limit)
    data = []
    async for doc in cursor:
        doc.pop("_id", None)
        if isinstance(doc.get("timestamp"), datetime):
            doc["timestamp"] = doc["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        data.append(doc)
    return data

# ----------------------------
# Gemini API Call
# ----------------------------
async def call_gemini_api(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "⚠️ Missing GEMINI_API_KEY"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return "AI service unavailable now."

# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, password: str):
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return templates.TemplateResponse("admin_dashboard.html", {"request": request})

@app.get("/api/admin/stats")
async def api_admin_stats():
    logs = await get_all_chat_logs(limit=2000)
    total = len(logs)
    unique = len(set(l.get("session_id") for l in logs))
    fallback = sum(1 for l in logs if l.get("intent") in ["fallback", "unknown"])
    rate = round((fallback / total) * 100, 2) if total else 0
    return JSONResponse({
        "total_messages": total,
        "unique_sessions": unique,
        "fallback_rate": rate,
        "recent_logs": logs[:10]
    })

# ----------------------------
# WebSocket Chat
# ----------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session = str(uuid.uuid4())
    print(f"⚡ WebSocket started: {session}")

    greeting = "Hello — I'm Humongous AI!"
    if engine:
        try:
            greeting = engine.get_response(engine.get_intent("hello"))
        except Exception:
            pass

    await ws.send_json({"type": "chat", "message": greeting})
    await log_interaction(session, "bot", greeting, intent="hello")

    try:
        while True:
            msg = await ws.receive_text()
            print(f"🗣 User: {msg}")
            await log_interaction(session, "user", msg)

            matched = engine.get_intent(msg)
            intent_tag = matched.get("tag", "unknown") if matched else "unknown"

            COMPLEX = {
                "creator", "creator_details", "who_are_you", "capabilities", "company_info",
                "hours", "location", "origin", "payments", "returns", "shipping", "tracking",
                "order_management", "discounts", "technical_support", "account_issues",
                "privacy_policy", "product_info", "feedback", "human_handoff", "billing_issues"
            }

            if intent_tag in COMPLEX:
                context = engine.get_response(matched)
                prompt = f"""
You are Humongous AI, a friendly assistant created by Akilan S R.
Answer using only this context: "{context}"
User asked: "{msg}"
"""
                bot_reply = f"✨ {await call_gemini_api(prompt)}"
            else:
                bot_reply = engine.get_response(matched)

            await ws.send_json({"type": "chat", "message": bot_reply})
            await log_interaction(session, "bot", bot_reply, response=bot_reply, intent=intent_tag)
    except WebSocketDisconnect:
        print(f"🔌 WebSocket closed: {session}")
    except Exception as e:
        print(f"⚠️ WebSocket Error: {e}")
        await ws.close()

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)