
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

from pymongo import MongoClient
from nlp_engine import NlpEngine

# ----------------------------
# CONFIG
# ----------------------------
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
# MONGO (SYNC)
# ----------------------------
if not MONGO_URI:
    raise RuntimeError("MONGO_URI missing in environment")

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
    print(f"❌ Failed to init NLP Engine: {e}")
    engine = None

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def log_interaction(session_id, sender, message, response=None, intent=None):
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
        print(f"✅ Logged chat for {sender}")
    except Exception as e:
        print(f"⚠️ Failed to log chat: {e}")

def get_all_chat_logs(limit=0) -> List[Dict[str, Any]]:
    # 0 = no limit (return all)
    cursor = (
        chat_collection.find().sort("timestamp", -1).limit(limit if limit > 0 else 100000)
        if chat_collection
        else []
    )
    logs = []
    for d in cursor:
        d.pop("_id", None)
        ts = d.get("timestamp")
        if isinstance(ts, datetime):
            d["timestamp"] = ts.strftime("%Y-%m-%d %H:%M:%S")
        logs.append(d)
    return logs

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
        print(f"❌ Gemini API error: {e}")
        return "AI service temporarily unavailable."

# ----------------------------
# ROUTES
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
async def admin_stats():
    logs = get_all_chat_logs(limit=0)  # return all messages
    total = len(logs)
    unique_sessions = len(set(l["session_id"] for l in logs))
    fallback_count = sum(1 for l in logs if l.get("intent") in ["fallback", "unknown"])
    fallback_rate = round((fallback_count / total) * 100, 2) if total else 0.0

    return JSONResponse({
        "total_messages": total,
        "unique_sessions": unique_sessions,
        "fallback_rate": fallback_rate,
        "recent_logs": logs  # full logs for admin dashboard
    })

# ----------------------------
# WEBSOCKET CHAT
# ----------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    print(f"⚡ WebSocket connected (session={session_id})")

    greeting = "Hello! I'm Humongous AI."
    if engine:
        try:
            greeting = engine.get_response(engine.get_intent("hello"))
        except Exception:
            pass

    await websocket.send_json({"type": "chat", "message": greeting})
    log_interaction(session_id, "bot", greeting, intent="hello")

    try:
        while True:
            user_text = await websocket.receive_text()
            print(f"🗣 User: {user_text}")
            log_interaction(session_id, "user", user_text)

            matched = engine.get_intent(user_text)
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
User asked: "{user_text}"
"""
                bot_text = f"✨ {await call_gemini_api(prompt)}"
            else:
                bot_text = engine.get_response(matched)

            await websocket.send_json({"type": "chat", "message": bot_text})
            log_interaction(session_id, "bot", bot_text, response=bot_text, intent=intent_tag)

    except WebSocketDisconnect:
        print(f"🔌 Disconnected (session={session_id})")
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")
        await websocket.close()

# ----------------------------
# START
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)