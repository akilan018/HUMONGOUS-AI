# main.py
# FINAL ULTIMATE VERSION: No database, Gemini for general questions, stable deployment.

import uuid
import httpx
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from nlp_engine import NlpEngine
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware 

# --- CONFIGURATION ---
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-latest:generateContent"

# --- INITIALIZATION ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None
try:
    engine = NlpEngine(intents_file="intents.json")
    print("✅ NLP Engine initialized successfully.")
except Exception as e:
    print(f"❌ Failed to initialize the NLP Engine: {e}")

# --- Gemini API Call Function ---
async def call_gemini_api(prompt: str):
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        return "**Error:** The `GEMINI_API_KEY` is not configured on the server."
    
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}], "tools": [{"google_search": {}}]}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            if "candidates" in result and result["candidates"]:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            return "I'm sorry, I couldn't generate a response at this time."
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return "I'm having trouble connecting to my extended knowledge base right now."

# --- HTTP ENDPOINT ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- WEBSOCKET ENDPOINT (FINAL HYBRID LOGIC) ---
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
            
            if intent_tag == 'fallback':
                print(f"Intent is 'fallback'. Calling Gemini API...")
                gemini_response = await call_gemini_api(user_message)
                bot_response = f"✨ {gemini_response}"
            else:
                bot_response = engine.get_response(matched_intent)
            
            await websocket.send_json({"type": "chat", "message": bot_response})
            print(f"==> Final Response Sent with Intent: '{intent_tag}'")

    except WebSocketDisconnect:
        print("WebSocket connection closed.")
    except Exception as e:
        print(f"An error occurred in WebSocket: {e}")
        await websocket.close(code=1011)
