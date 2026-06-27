from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ml.api.engine import ChatbotEngine
from ml.api.response_builder import build_response

# Inisialisasi API dan Engine
app = FastAPI(title="Chatbot NLP Engine Palembang")

# Tambahkan CORS Middleware agar bisa diakses dari React Native (Expo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Mengizinkan akses dari mana saja (sementara untuk dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = ChatbotEngine()

# Definisi format request
class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "NLP Engine is running!"}

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    # Proses teks pengguna menggunakan model
    result = engine.process_message(request.message)
    
    intent = result["intent"]
    entities = result["entities"]
    
    # Dapatkan jawaban natural (query Supabase di dalam sini jika diperlukan)
    reply_text = build_response(intent, entities)
    
    # Format respon yang akan dikembalikan ke React Native
    response_data = {
        "status": "success",
        "data": result,
        "reply": reply_text
    }

    return response_data