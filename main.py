from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# --------------------
# CONFIGURATION
# --------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT").rstrip("/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION = os.getenv("API_VERSION")  
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")  

client = AzureOpenAI(
    api_version=API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)

# --------------------
# FASTAPI APP
# --------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Session Memory
# --------------------
sessions = {}  # key: session_id, value: messages list

SYSTEM_PROMPT = {"role": "system", "content": "You are Chaaya, calm AI guide for BODH."}
MAX_MESSAGES = 15

# --------------------
# Request Model
# --------------------
class ChatRequest(BaseModel):
    session_id: str
    message: str
    end_session: bool = False

# --------------------
# Chat Endpoint
# --------------------
@app.post("/chat")
async def chat_with_assistant(body: ChatRequest):
    session_id = body.session_id

    
    if body.end_session:
        sessions.pop(session_id, None)
        return {"reply": "Session ended. All memory cleared."}

    
    if session_id not in sessions:
        sessions[session_id] = [SYSTEM_PROMPT]

    session_messages = sessions[session_id]

   
    if len(session_messages) >= MAX_MESSAGES:
        return {"reply": "Session limit reached. Start a new session."}

    
    session_messages.append({"role": "user", "content": body.message})

    
    response = client.chat.completions.create(
        messages=session_messages,
        model=DEPLOYMENT_NAME,
        max_completion_tokens=1024,   
    )

    reply = response.choices[0].message.content

   
    session_messages.append({"role": "assistant", "content": reply})

    return {"reply": reply}
