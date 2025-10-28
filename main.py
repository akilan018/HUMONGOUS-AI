import uuid
import httpx
import os
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pymongo import MongoClient
from nlp_engine import NlpEngine

# === Load environment variables ===
load_dotenv()

# --- GEMINI CONFIG ---
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-latest:generateContent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- MONGODB CONFIG ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://akilan200518_db_user:KPtCEpdfiXOYqCm9@cluster3.3rpkllm.mongodb.net/humongous_ai?retryWrites=true&w=majority&appName=Cluster3")

# --- INITIALIZATION ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Allow frontend communication (Render & local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === MongoDB Connection ===
try:
    client = MongoClient(MONGO_URI)
    db = client["humongous_ai"]
    chat_collection = db["chat_logs"]
    print("✅ Connected to MongoDB Atlas successfully.")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    db = None
    chat_collection = None

# === NLP Engine Initialization ===
try:
    engine = NlpEngine(intents_file="intents.json")
    print("✅ NLP Engine initialized successfully.")
except Exception as e:
    print(f"❌ NLP Engine initialization failed: {e}")
    engine = None


# --- GEMINI CALL FUNCTION ---
async def call_gemini_api(prompt: str):
    if not GEMINI_API_KEY:
        return "**Error:** Missing GEMINI_API_KEY."
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", json=payload, headers=headers
            )
            response.raise_for_status()
            result = response.json()
            if "candidates" in result and result["candidates"]:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            return "I'm sorry, I couldn't generate a unique response at this time."
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return "I'm having trouble connecting to my AI service right now."


# --- ROOT ENDPOINT ---
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# --- WEBSOCKET CHAT HANDLER ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    print(f"💬 WebSocket connected (session {session_id})")

    if not engine:
        await websocket.send_json({"type": "error", "message": "NLP engine not available."})
        await websocket.close()
        return

    # Greeting message
    greeting = engine.get_response(engine.get_intent("hello"))
    await websocket.send_json({"type": "chat", "message": greeting})

    try:
        while True:
            user_message = await websocket.receive_text()
            print(f"🗣️ User: {user_message}")

            # Log user message
            if chat_collection:
                chat_collection.insert_one({
                    "session_id": session_id,
                    "timestamp": datetime.utcnow(),
                    "sender": "user",
                    "message": user_message
                })

            matched_intent = engine.get_intent(user_message)
            intent_tag = matched_intent.get("tag", "unknown") if matched_intent else "unknown"

            COMPLEX_INTENTS = [
                'creator', 'creator_details', 'who_are_you', 'capabilities', 'company_info',
                'hours', 'location', 'origin', 'payments', 'returns', 'shipping', 'tracking',
                'order_management', 'discounts', 'technical_support', 'account_issues',
                'privacy_policy', 'product_info', 'feedback', 'human_handoff', 'billing_issues'
            ]

            # --- Handle intents ---
            if intent_tag in COMPLEX_INTENTS:
                print(f"⚙️ Complex intent '{intent_tag}', calling Gemini API...")

                context_answer = engine.get_response(matched_intent)
                prompt = f"""
                You are Humongous AI, a friendly and helpful assistant created by Akilan S R.
                Use ONLY the context below to answer conversationally and informatively.

                CONTEXT:
                "{context_answer}"

                USER MESSAGE:
                "{user_message}"

                DETAILED ANSWER:
                """

                gemini_response = await call_gemini_api(prompt)
                bot_response = f"✨ {gemini_response}"

            else:
                print(f"⚡ Simple intent '{intent_tag}' handled locally.")
                bot_response = engine.get_response(matched_intent)

            # Log bot response
            if chat_collection:
                chat_collection.insert_one({
                    "session_id": session_id,
                    "timestamp": datetime.utcnow(),
                    "sender": "bot",
                    "message": bot_response,
                    "intent": intent_tag
                })

            await websocket.send_json({"type": "chat", "message": bot_response})

    except WebSocketDisconnect:
        print(f"🔌 Session {session_id} disconnected.")
    except Exception as e:
        print(f"❌ Error in WebSocket: {e}")
        await websocket.close(code=1011)