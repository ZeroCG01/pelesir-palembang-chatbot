from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, Any]]] = []

class ActionButton(BaseModel):
    type: str
    label: str
    destination_id: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    origin: Optional[str] = None
    destination: Optional[str] = None

class DestinationCard(BaseModel):
    id: str
    name: str
    image_url: Optional[str] = None
    rating: Optional[float] = None
    category: Optional[str] = None
    price_text: Optional[str] = None

class QuickReply(BaseModel):
    label: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    source: Optional[str] = "local"
    actions: Optional[List[ActionButton]] = None
    cards: Optional[List[DestinationCard]] = None
    quick_replies: Optional[List[QuickReply]] = None
