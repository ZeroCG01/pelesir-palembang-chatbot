from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.nlp_service import nlp_model

router = APIRouter()

@router.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Pesan tidak boleh kosong")
            
        # Panggil NLP service untuk mendapatkan balasan
        reply_text = nlp_model.generate_reply(request.message, request.history)
        
        return ChatResponse(reply=reply_text)
    except HTTPException:
        raise  # Re-raise HTTPException agar tidak dibungkus ulang jadi 500
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
