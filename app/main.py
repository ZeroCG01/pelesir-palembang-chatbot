from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chat

app = FastAPI(
    title="Pelesir Palembang Chatbot API",
    description="API untuk NLP Chatbot Pelesir Palembang",
    version="1.0.0"
)

# Konfigurasi CORS agar React Native (baik via Web, Android, iOS) bisa mengakses
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Bisa diganti dengan origin spesifik jika sudah production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check Endpoint (dipakai oleh chatbotService.checkConnection)
@app.get("/")
async def root():
    return {"status": "ok", "message": "Chatbot Service is running"}

# Include routers
app.include_router(chat.router)
