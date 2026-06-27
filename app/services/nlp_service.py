from ml.api.engine import ChatbotEngine
from ml.api.response_builder import build_response, ABBREVIATIONS

class ChatbotModel:
    def __init__(self):
        print("Mempersiapkan model PyTorch/Transformers dari folder ml/saved_models...")
        self.engine = ChatbotEngine()

    def generate_reply(self, message: str) -> str:
        msg_lower = message.lower()
        
        # --- RULE-BASED INTERCEPTOR ---
        # 1. Intercept Itinerary
        if "hari" in msg_lower and ("rekomendasi" in msg_lower or "itinerary" in msg_lower or "wisata" in msg_lower or "jadwal" in msg_lower):
            days = 1
            if "2 hari" in msg_lower or "dua hari" in msg_lower: days = 2
            elif "3 hari" in msg_lower or "tiga hari" in msg_lower: days = 3
            return build_response("rule_itinerary", {"DAYS": days})
            
        # 2. Intercept Hotel/Akomodasi
        if "hotel" in msg_lower or "penginapan" in msg_lower or "menginap" in msg_lower:
            daerah = ""
            murah = "murah" in msg_lower
            mahal = "mahal" in msg_lower or "mewah" in msg_lower
            if "daerah " in msg_lower:
                daerah = msg_lower.split("daerah ")[1].strip()
            elif "dekat " in msg_lower:
                daerah = msg_lower.split("dekat ")[1].strip()
            elif "di " in msg_lower:
                daerah = msg_lower.split("di ")[1].strip()
            return build_response("rule_hotel", {"DAERAH": daerah, "MURAH": murah, "MAHAL": mahal})

        # 1. Klasifikasi Intent & Ekstraksi Entitas dari user message
        result = self.engine.process_message(message)
        print(f"ML Result: {result}")
        
        entities = result["entities"]
        
        # --- FALLBACK HEURISTIC ---
        # Jika model Machine Learning gagal mendeteksi entitas DESTINATION,
        # kita lakukan pengecekan manual berdasarkan Kamus Singkatan (ABBREVIATIONS)
        if "DESTINATION" not in entities:
            msg_lower = f" {message.lower()} "
            for short_name in ABBREVIATIONS.keys():
                if f" {short_name} " in msg_lower:
                    entities["DESTINATION"] = short_name
                    print(f"Fallback NER: Found '{short_name}' manually.")
                    break
        
        # 2. Hasilkan balasan natural (melibatkan database Supabase di dalam builder)
        reply_text = build_response(result["intent"], entities)
        
        return reply_text

# Instansiasi global agar model hanya di-load sekali ke memory
nlp_model = ChatbotModel()

