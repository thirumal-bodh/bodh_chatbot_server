from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()


# --------------------
# CONFIGURATION
# --------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT").rstrip("/")  # Remove trailing slash
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION = os.getenv("API_VERSION")
AZURE_ASSISTANT_ID = os.getenv("AZURE_ASSISTANT_ID")

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
# Request Model
# --------------------
class ChatRequest(BaseModel):
    message: str

# --------------------
# Chat Endpoint
# --------------------
@app.post("/chat")
async def chat_with_assistant(body: ChatRequest):
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }

    # 1. Create thread
    thread_resp = requests.post(
        f"{AZURE_OPENAI_ENDPOINT}/openai/threads?api-version={API_VERSION}",
        headers=headers,
    )
    thread_resp.raise_for_status()
    thread_id = thread_resp.json()["id"]


    user_message = body.message

    msg_resp = requests.post(
        f"{AZURE_OPENAI_ENDPOINT}/openai/threads/{thread_id}/messages?api-version={API_VERSION}",
        headers=headers,
        json={
            "role": "user",
            "content": user_message
        },
    )
    msg_resp.raise_for_status()

    # 3. Create run
    run_resp = requests.post(
        f"{AZURE_OPENAI_ENDPOINT}/openai/threads/{thread_id}/runs?api-version={API_VERSION}",
        headers=headers,
        json={
            "assistant_id": AZURE_ASSISTANT_ID
        }
    )
    run_resp.raise_for_status()
    run_id = run_resp.json()["id"]

    # 4. Poll until run completes
    while True:
        run_status_resp = requests.get(
            f"{AZURE_OPENAI_ENDPOINT}/openai/threads/{thread_id}/runs/{run_id}?api-version={API_VERSION}",
            headers=headers,
        )
        run_status_resp.raise_for_status()
        run_status = run_status_resp.json()["status"]

        if run_status == "completed":
            break
        elif run_status in ["failed", "cancelled"]:
            return {"error": f"Run failed with status {run_status}"}

        await asyncio.sleep(1)

    # 5. Retrieve assistant messages
    messages_resp = requests.get(
        f"{AZURE_OPENAI_ENDPOINT}/openai/threads/{thread_id}/messages?api-version={API_VERSION}",
        headers=headers,
    )
    messages_resp.raise_for_status()
    messages = messages_resp.json()["data"]

    assistant_reply = "No reply found."
    for m in reversed(messages):
        if m["role"] == "assistant":
            assistant_reply = m["content"][0]["text"]["value"]
            break

    return {
        "reply": assistant_reply  
    }
