import os
import google.generativeai as genai
from ml.api.engine import ChatbotEngine
from ml.api.response_builder import build_response, ABBREVIATIONS

# Konfigurasi Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

SYSTEM_PROMPT = """Anda adalah TanyaKito, asisten virtual ramah untuk aplikasi Pelesir Palembang.
Tugas Anda adalah menjawab pertanyaan pengguna seputar wisata Palembang berdasarkan riwayat percakapan.
Fakta Penting: Palembang adalah kota berbasis sungai (Sungai Musi), BUKAN kota pesisir/pantai. Jika pengguna bertanya tentang "pantai" di Palembang, beritahu dengan sopan bahwa Palembang tidak memiliki pantai laut, dan sarankan alternatif wisata air seperti menyusuri Sungai Musi, mengunjungi Pulau Kemaro, atau wisata alam lainnya.
Gunakan bahasa Indonesia yang ramah, sopan, sedikit santai, dan informatif. Anda boleh menyisipkan sedikit bahasa daerah Palembang jika relevan, tapi utamakan bahasa Indonesia yang mudah dipahami.
"""

class ChatbotModel:
    def __init__(self):
        print("Mempersiapkan model PyTorch/Transformers dari folder ml/saved_models...")
        self.engine = ChatbotEngine()
        self.gemini_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=SYSTEM_PROMPT
        )

    def generate_gemini_reply(self, message: str, history: list) -> str:
        try:
            print("Fallback to Gemini LLM with context...")
            # Konversi format history ke format yang diterima SDK Gemini
            gemini_history = []
            for h in history:
                role = h.get("role", "user")
                # Pastikan role sesuai standar Gemini ('user' atau 'model')
                if role not in ["user", "model"]:
                    role = "user"
                gemini_history.append({
                    "role": role,
                    "parts": [h.get("content", "")]
                })
            
            chat = self.gemini_model.start_chat(history=gemini_history)
            response = chat.send_message(message)
            return response.text
        except Exception as e:
            print(f"Gemini Fallback Error: {e}")
            return "Maaf, saya tidak mengerti maksud Anda. Silakan coba tanyakan hal lain seputar wisata Palembang."

    def generate_reply(self, message: str, history: list = None) -> str:
        if history is None:
            history = []
            
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
        intent = result["intent"]
        
        # --- FALLBACK HEURISTIC LOKAL ---
        if "DESTINATION" not in entities:
            msg_lower_padded = f" {message.lower()} "
            for short_name in ABBREVIATIONS.keys():
                if f" {short_name} " in msg_lower_padded:
                    entities["DESTINATION"] = short_name
                    print(f"Fallback NER: Found '{short_name}' manually.")
                    break
        
        # 2. Hasilkan balasan natural (melibatkan database Supabase di dalam builder)
        reply_text = build_response(intent, entities)
        
        # 3. TRIGGER GEMINI FALLBACK
        # Jika model lokal gagal memberikan informasi spesifik atau menghasilkan pesan error default
        if "Maaf, saya tidak mengerti" in reply_text or "Maaf, saya tidak menemukan tempat wisata" in reply_text or intent == "ask_unrelated":
            return self.generate_gemini_reply(message, history)
        
        # Jika bukan error, kembalikan jawaban dari ML Lokal
        return reply_text

# Instansiasi global agar model hanya di-load sekali ke memory
nlp_model = ChatbotModel()

