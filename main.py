import uuid
import httpx
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pymongo import MongoClient

from nlp_engine import NlpEngine  # your existing NLP engine

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# --- GEMINI CONFIG ---
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-latest:generateContent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- MONGODB CONFIG ---
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["humongous_ai"]
chat_logs = db["chat_logs"]

# --- FASTAPI SETUP ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Allow frontend to connect from anywhere (important for Render)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- NLP ENGINE INITIALIZATION ---
engine = None
try:
    engine = NlpEngine(intents_file="intents.json")
    print("✅ NLP Engine initialized successfully.")
except Exception as e:
    print(f"❌ Failed to initialize the NLP Engine: {e}")

# --- GEMINI API FUNCTION ---
async def call_gemini_api(prompt: str):
    if not GEMINI_API_KEY:
        return "**Error:** GEMINI_API_KEY not configured on server."
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            if "candidates" in result and result["candidates"]:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            return "I'm sorry, I couldn't generate a unique response right now."
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return "I'm having trouble connecting to the AI service right now."

# --- STORE CHAT LOG ---
def log_chat(user_message, bot_response, intent):
    try:
        chat_logs.insert_one({
            "user_message": user_message,
            "bot_response": bot_response,
            "intent": intent
        })
        print("✅ Chat logged successfully.")
    except Exception as e:
        print(f"⚠️ Failed to log chat: {e}")

# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- WEBSOCKET (CHAT LOGIC) ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("⚡ WebSocket connected.")

    if not engine:
        await websocket.send_json({"type": "error", "message": "NLP engine not available."})
        await websocket.close()
        return

    greeting = engine.get_response(engine.get_intent("hello"))
    await websocket.send_json({"type": "chat", "message": greeting})

    try:
        while True:
            user_message = await websocket.receive_text()
            matched_intent = engine.get_intent(user_message)
            intent_tag = matched_intent.get('tag', 'unknown') if matched_intent else 'unknown'

            COMPLEX_INTENTS = [
                'creator', 'creator_details', 'who_are_you', 'capabilities', 'company_info',
                'hours', 'location', 'origin', 'payments', 'returns', 'shipping', 'tracking',
                'order_management', 'discounts', 'technical_support', 'account_issues',
                'privacy_policy', 'product_info', 'feedback', 'human_handoff', 'billing_issues'
            ]

            # --- GENERATIVE OR STATIC RESPONSE ---
            if intent_tag in COMPLEX_INTENTS:
                print(f"💡 Intent '{intent_tag}' → Gemini API mode.")
                context_answer = engine.get_response(matched_intent)
                prompt = f"""
                You are Humongous AI, created by Akilan S R.
                Be helpful, polite, and professional.
                Use only the following context to answer:

                CONTEXT:
                "{context_answer}"

                USER:
                "{user_message}"
                """
                bot_response = await call_gemini_api(prompt)
            else:
                print(f"💬 Intent '{intent_tag}' → Static response mode.")
                bot_response = engine.get_response(matched_intent)

            # --- SAVE TO MONGODB ---
            log_chat(user_message, bot_response, intent_tag)

            await websocket.send_json({"type": "chat", "message": bot_response})
            print(f"✅ Sent reply for intent: {intent_tag}")

    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected.")
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")
        await websocket.close(code=1011)