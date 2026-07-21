from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, Any]]] = []

class ChatResponse(BaseModel):
    reply: str
    source: Optional[str] = "local"
