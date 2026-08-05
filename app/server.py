import os
import sys
import queue
import threading
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Add project root to path so we can import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.scratch_model import train_scratch_model, load_saved_scratch_model, DEVICE
from models.pretrained_model import PretrainedCodeGenerator
from models.gemini_chatbot import generate_chat_response
from typing import List, Optional

app = FastAPI(title="AI Code Architect Dashboard")

# Ensure static and template folders exist
os.makedirs("app/static/css", exist_ok=True)
os.makedirs("app/static/js", exist_ok=True)
os.makedirs("app/templates", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Pretrained generator instance
pretrained_gen = PretrainedCodeGenerator()

class PromptRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 256

class ScratchGenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 150
    temperature: float = 0.7

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/api/generate/pretrained")
async def generate_pretrained(req: PromptRequest):
    try:
        # Generate code asynchronously in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        code = await loop.run_in_executor(
            None, 
            pretrained_gen.generate, 
            req.prompt, 
            req.max_tokens, 
            req.temperature
        )
        return {"status": "success", "code": code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/scratch")
async def generate_scratch(req: ScratchGenerateRequest):
    try:
        # Load the latest custom trained model if it exists
        model, tokenizer = load_saved_scratch_model(os.path.abspath("data"))
        if model is None or tokenizer is None:
            return {
                "status": "error", 
                "message": "Custom model has not been trained yet. Please train it in the 'Train Scratch Model' tab first!"
            }
            
        import torch
        context_tokens = tokenizer.encode(req.prompt)
        if not context_tokens:
            context_tokens = tokenizer.encode("def ")
            
        context = torch.tensor([context_tokens], dtype=torch.long, device=DEVICE)
        
        loop = asyncio.get_event_loop()
        generated_ids = await loop.run_in_executor(
            None,
            model.generate,
            context,
            req.max_tokens,
            req.temperature
        )
        
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        return {"status": "success", "code": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ChatMessage(BaseModel):
    role: str
    text: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage]
    apiKey: Optional[str] = None
    modelName: Optional[str] = "gemini-1.5-flash"

@app.post("/api/chat")
async def chat_gemini(req: ChatRequest):
    try:
        api_key = req.apiKey or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {
                "status": "error",
                "message": "Gemini API Key is not set. Please provide it in the chatbot settings panel."
            }
            
        loop = asyncio.get_event_loop()
        history_list = [{"role": m.role, "text": m.text} for m in req.history]
        
        reply = await loop.run_in_executor(
            None,
            generate_chat_response,
            api_key,
            req.message,
            history_list,
            req.modelName
        )
        return {"status": "success", "reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/train")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Queue for inter-thread communication
    msg_queue = queue.Queue()
    
    # Callback run in the PyTorch training thread
    def progress_cb(epoch, train_loss, val_loss, sample_text):
        msg_queue.put({
            "type": "progress",
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "sample": sample_text
        })
        
    try:
        # Read the training config from websocket
        config = await websocket.receive_json()
        epochs = int(config.get("epochs", 500))
        lr = float(config.get("lr", 0.001))
        batch_size = int(config.get("batch_size", 32))
        
        dataset_path = os.path.abspath("data/python_dataset.txt")
        
        # Thread worker function
        def worker():
            try:
                train_scratch_model(
                    dataset_path=dataset_path,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    progress_callback=progress_cb
                )
                msg_queue.put({"type": "complete"})
            except Exception as ex:
                msg_queue.put({"type": "error", "message": str(ex)})
                
        # Start training thread
        thread = threading.Thread(target=worker)
        thread.start()
        
        # Poll queue and send updates to websocket
        while thread.is_alive() or not msg_queue.empty():
            try:
                msg = msg_queue.get_nowait()
                await websocket.send_json(msg)
                if msg["type"] in ["complete", "error"]:
                    break
            except queue.Empty:
                await asyncio.sleep(0.05)
                
    except WebSocketDisconnect:
        print("Training websocket disconnected by client.")
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
