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
# DATABASE SETUP
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
if not db or db.name == "admin":
    db = mongo_client["humongous_ai"]

chat_collection = db["chat_logs"]

# ----------------------------
# NLP ENGINE INIT
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
    """Convert any timestamp to readable format."""
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(ts, str):
        return ts
    return str(ts)


def log_interaction(session_id: str, sender: str, message: str, response: str = None, intent: str = None):
    """Log conversation messages into MongoDB."""
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
        print(f"✅ Log saved for {sender}")
    except Exception as e:
        print(f"⚠️ Failed to log chat: {e}")


def get_all_chat_logs(limit: int = 0) -> List[Dict[str, Any]]:
    """Retrieve logs safely for dashboard."""
    try:
        cursor = chat_collection.find().sort("timestamp", -1)
        if limit > 0:
            cursor = cursor.limit(limit)
        logs = []
        for doc in cursor:
            doc.pop("_id", None)
            doc["timestamp"] = _format_timestamp(doc.get("timestamp"))
            logs.append(doc)
        return logs
    except Exception as e:
        print(f"⚠️ Failed to get logs: {e}")
        return []


async def call_gemini_api(prompt: str) -> str:
    """Query Gemini API — restricted to customer FAQs only."""
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
                return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
    return "I'm only trained to answer questions about our services and support."

# ----------------------------
# ROUTES
# ----------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main chatbot UI"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, password: str):
    """Admin panel with password gate"""
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return templates.TemplateResponse("admin_dashboard.html", {"request": request})


@app.get("/api/admin/stats")
def admin_stats():
    """Dashboard analytics data endpoint"""
    try:
        logs = get_all_chat_logs()
        total_messages = len(logs)
        unique_sessions = len({l.get("session_id") for l in logs if l.get("session_id")})
        fallback_count = sum(1 for l in logs if l.get("intent") in ["fallback", "unknown"])
        fallback_rate = round((fallback_count / total_messages) * 100, 2) if total_messages else 0.0

        # Intent distribution
        intent_counts: Dict[str, int] = {}
        for l in logs:
            intent = l.get("intent", "unknown")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        # Convert for charts
        chart_labels = list(intent_counts.keys())
        chart_values = list(intent_counts.values())

        # Recent chat logs
        recent_logs = [
            {
                "timestamp": l.get("timestamp"),
                "user_message": l.get("message", "") if l.get("sender") == "user" else "",
                "intent_detected": l.get("intent", "")
            }
            for l in logs
            if l.get("sender") == "user"
        ]

        analytics = {
            "total_messages": total_messages,
            "unique_sessions": unique_sessions,
            "fallback_rate": f"{fallback_rate}%",
            "raw_fallback_rate": fallback_rate,
        }

        return JSONResponse({
            "analytics": analytics,
            "logs": recent_logs,
            "chart_labels": chart_labels,
            "chart_values": chart_values
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
    await websocket.send_json({"type": "chat", "message": greeting})
    log_interaction(session_id, "bot", greeting, intent="hello")

    try:
        while True:
            user_msg = await websocket.receive_text()
            print(f"🗣 User: {user_msg}")
            log_interaction(session_id, "user", user_msg)

            intent_tag = "unknown"
            matched_intent = None

            if engine:
                try:
                    matched_intent = engine.get_intent(user_msg)
                    intent_tag = matched_intent.get("tag", "unknown")
                except Exception:
                    intent_tag = "unknown"

            if engine and intent_tag not in ["fallback", "unknown"]:
                try:
                    bot_reply = engine.get_response(matched_intent)
                except Exception:
                    bot_reply = "Sorry, I couldn’t find that answer right now."
            else:
                # Use Gemini for company-related FAQs only
                company_context = (
                    "You are Humongous AI, a professional assistant for Akilan S R's company. "
                    "Only answer company-related FAQs such as services, support, pricing, and policies. "
                    "If asked general or unrelated questions, respond: "
                    "'I'm only trained to answer questions about our services and support.'"
                )
                prompt = f"{company_context}\nUser: {user_msg}\nProvide a clear, short answer."
                bot_reply = await call_gemini_api(prompt)

            await websocket.send_json({"type": "chat", "message": bot_reply})
            log_interaction(session_id, "bot", bot_reply, response=bot_reply, intent=intent_tag)

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