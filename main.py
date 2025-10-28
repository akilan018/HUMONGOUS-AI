# main.py

import uuid
import httpx
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from nlp_engine import NlpEngine
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware 
from pymongo import MongoClient  # ✅ MongoDB support

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# --- CONFIGURATION ---
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-latest:generateContent"
MONGO_URI = os.getenv("MONGO_URI")

# --- INITIALIZATION ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- CORS FIX (For deployment / local testing) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONNECT TO MONGODB ---
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client["humongous_ai"]  # ✅ your database name
    chat_logs = db["chat_logs"]        # collection name
    print("✅ MongoDB connected successfully.")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    db = None

# --- NLP ENGINE INITIALIZATION ---
engine = None
try:
    engine = NlpEngine(intents_file="intents.json")
    print("✅ NLP Engine initialized successfully.")
except Exception as e:
    print(f"❌ Failed to initialize the NLP Engine: {e}")

# --- GEMINI API CALL FUNCTION ---
async def call_gemini_api(prompt: str):
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        return "**Error:** The `GEMINI_API_KEY` is not configured on the server."
    
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            if "candidates" in result and result["candidates"]:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            return "I'm sorry, I couldn't generate a unique response at this time."
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return "I'm having trouble connecting to my generative AI service right now."

# --- HTTP ENDPOINT ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- WEBSOCKET ENDPOINT ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established.")
    
    if engine:
        initial_greeting = engine.get_response(engine.get_intent("hello"))
        await websocket.send_json({"type": "chat", "message": initial_greeting})
    else:
        await websocket.send_json({"type": "error", "message": "The NLP engine is not available."})
        await websocket.close(); return
        
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
            
            if intent_tag in COMPLEX_INTENTS:
                # For complex business intents, use Gemini API for detailed response
                print(f"Intent '{intent_tag}' → Using Gemini API")
                
                context_answer = engine.get_response(matched_intent)
                
                prompt = f"""
                You are Humongous AI, a friendly and knowledgeable assistant created by Akilan S R.
                Your goal is to give a clear, conversational answer based ONLY on the provided context.
                Do not add unrelated details.

                CONTEXT:
                "{context_answer}"

                USER QUESTION:
                "{user_message}"

                ANSWER:
                """
                
                gemini_response = await call_gemini_api(prompt)
                bot_response = f"✨ {gemini_response}"

            else:
                # For simple or fallback intents, use static response
                print(f"Intent '{intent_tag}' → Using static response")
                bot_response = engine.get_response(matched_intent)
            
            # ✅ Store chat in MongoDB
            if db:
                chat_logs.insert_one({
                    "session_id": str(uuid.uuid4()),
                    "user_message": user_message,
                    "bot_response": bot_response,
                    "intent": intent_tag
                })

            await websocket.send_json({"type": "chat", "message": bot_response})
            print(f"==> Final Response Sent ({intent_tag})")

    except WebSocketDisconnect:
        print("WebSocket connection closed.")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        await websocket.close(code=1011)