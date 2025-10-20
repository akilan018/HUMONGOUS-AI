# main.py
# FINAL VERSION: This version is restricted to ONLY answer questions from intents.json.

import uuid
import httpx
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from nlp_engine import NlpEngine
from database import init_db, log_interaction, get_chat_history, get_all_chat_logs, IS_DB_ACTIVE
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware 
from collections import Counter

# --- CONFIGURATION ---
ADMIN_PASSWORD = "MySuperSecretPassword123" 
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
    init_db() 
except Exception as e:
    print(f"❌ Failed to initialize the application: {e}")

# --- Gemini API Call Function (Now only used for summarization) ---
async def call_gemini_api(prompt: str):
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        return "**Error:** The `GEMINI_API_KEY` is not found."
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            if "candidates" in result and result["candidates"]:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            return "I'm sorry, I couldn't generate a summary."
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return "Could not connect to the summarization service."

# --- HTTP ENDPOINTS ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def get_admin_dashboard(request: Request, password: str = ""):
    if password != ADMIN_PASSWORD: raise HTTPException(status_code=401, detail="Unauthorized")
    if not IS_DB_ACTIVE:
        return HTMLResponse("<h1>Database Feature Disabled</h1><p>The Admin Dashboard is unavailable because this application is running on a free tier without a persistent database.</p>", status_code=503)
    
    all_logs = get_all_chat_logs()
    total_messages = len(all_logs)
    unique_sessions = len(set(log['session_id'] for log in all_logs))
    fallback_count = sum(1 for log in all_logs if log['intent_detected'] == 'fallback')
    fallback_rate = (fallback_count / total_messages * 100) if total_messages > 0 else 0
    intent_counts = Counter(log['intent_detected'] for log in all_logs)
    chart_data = intent_counts.most_common(7)
    analytics = { "total_messages": total_messages, "unique_sessions": unique_sessions, "fallback_rate": f"{fallback_rate:.1f}%", "raw_fallback_rate": fallback_rate }
    
    return templates.TemplateResponse("admin.html", { "request": request, "logs": all_logs, "analytics": analytics, "chart_labels": [c[0] for c in chart_data], "chart_values": [c[1] for c in chart_data] })

@app.get("/history/{session_id}", response_class=JSONResponse)
async def get_history(session_id: str):
    return {"history": get_chat_history(session_id)}
    
@app.post("/summarize", response_class=JSONResponse)
async def summarize_history(request: Request):
    if not IS_DB_ACTIVE: return JSONResponse({"summary": "**Error:** History is disabled on the free plan."}, status_code=200)
    data = await request.json()
    transcript = data.get("transcript")
    prompt = f"Please provide a concise, bullet-point summary of this conversation:\n\n---\n{transcript}\n---"
    summary = await call_gemini_api(prompt)
    return {"summary": summary}

# --- WEBSOCKET ENDPOINT (CORRECTED AND RESTRICTED) ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = websocket.query_params.get("session_id") or str(uuid.uuid4())
    is_new_session = not websocket.query_params.get("session_id")

    if is_new_session: await websocket.send_json({"type": "session", "session_id": session_id})
    if is_new_session and engine:
        initial_greeting = engine.get_response(engine.get_intent("hello"))
        await websocket.send_json({"type": "chat", "message": initial_greeting})

    if not engine:
        await websocket.send_json({"type": "error", "message": "The NLP engine is not available."}); await websocket.close(); return
        
    try:
        while True:
            user_message = await websocket.receive_text()
            matched_intent = engine.get_intent(user_message)
            bot_response = engine.get_response(matched_intent)
            intent_tag = matched_intent.get('tag', 'unknown') if matched_intent else 'unknown'
            
            if IS_DB_ACTIVE:
                log_interaction(session_id, user_message, bot_response, intent_tag)
            
            await websocket.send_json({"type": "chat", "message": bot_response})
            print(f"==> Final Response Sent with Intent: '{intent_tag}'")

    except WebSocketDisconnect:
        print(f"WebSocket connection closed for Session ID: {session_id}")
    except Exception as e:
        print(f"An error occurred in WebSocket: {e}")
        await websocket.close(code=1011)
