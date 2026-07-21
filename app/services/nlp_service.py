import os
import time
import google.generativeai as genai
from ml.api.engine import ChatbotEngine
from ml.api.response_builder import build_response, ABBREVIATIONS, supabase

# Konfigurasi Gemini API (Rotasi Multi-Key & Fallback Paid Key)
raw_keys = os.environ.get("GEMINI_API_KEY", "")
free_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
paid_key = os.environ.get("GEMINI_API_KEY_PAID", "").strip()

all_api_keys = free_keys + ([paid_key] if paid_key and paid_key not in free_keys else [])
if not all_api_keys:
    all_api_keys = [""]  # Fallback agar tidak crash

current_key_idx = 0
genai.configure(api_key=all_api_keys[current_key_idx])

def build_system_prompt():
    base_prompt = """Anda adalah TanyaKito, asisten virtual ramah untuk aplikasi Pelesir Palembang.
Tugas Anda adalah menjawab pertanyaan pengguna seputar wisata Palembang berdasarkan riwayat percakapan.

ATURAN PENTING:
1. Palembang adalah kota berbasis sungai (Sungai Musi), BUKAN kota pesisir/pantai. Jika pengguna bertanya tentang "pantai" di Palembang, beritahu bahwa Palembang tidak memiliki pantai laut, dan sarankan alternatif seperti Sungai Musi atau Pulau Kemaro.
2. JANGAN mengulangi salam pembuka (seperti "Halo! Selamat datang!") jika pengguna sedang melakukan percakapan lanjutan. Langsung jawab intinya.
3. Singkatan "SMB" dalam konteks wisata Palembang merujuk pada "Museum Sultan Mahmud Badaruddin II", BUKAN bandara.
4. Jawablah seringkas dan sesantai mungkin. Gunakan bahasa Indonesia.
5. Jika pengguna meminta dibuatkan itinerary (jadwal perjalanan), susunlah jadwal yang masuk akal (Pagi, Siang, Sore) sesuai jumlah hari dan KATEGORI yang mereka minta (misal: wisata alam saja, kuliner saja, atau campuran). Pilihlah dari DATABASE di bawah ini.
6. Jika pengguna meminta rekomendasi wisata (baik menyebutkan kategori seperti 'alam' maupun tidak), berikan rekomendasi dalam bentuk DAFTAR (list). Pilihlah dari DATABASE di bawah ini.
7. KONTEKS TOPIK TERAKHIR SANGAT KRITIKAL: Jika pengguna menanyakan tempat lain dengan format lanjutan (misal: "kalau smb?", "gimana dengan ptc?", "kalo di ampera?"), kamu HARUS melihat pertanyaan pengguna sebelumnya. Jawab HANYA spesifik tentang topik yang sama.
- PERINTAH MUTLAK: Jika sebelumnya menanyakan harga tiket, WAJIB jawab harga tiketnya saja (maksimal 1-2 kalimat). DILARANG KERAS memberikan alamat, jam buka, deskripsi, atau informasi tambahan apa pun yang tidak diminta!
8. KONTEKS TEMPAT TERAKHIR: Jika pengguna bertanya tanpa menyebutkan nama tempat (misal: "jam bukanya?", "info detailnya?", "fasilitasnya apa aja?"), SELALU asumsikan mereka menanyakan TEMPAT WISATA TERAKHIR yang sedang dibahas. Jawab HANYA pertanyaan spesifik tersebut, dengan sangat ringkas.
9. JANGAN PERNAH menyuruh pengguna untuk melihat informasi/detail di "halaman beranda". Langsung berikan informasi dari database di bawah ini.
10. PENOLAKAN OUT-OF-DOMAIN (SANGAT PENTING): Jika pengguna memberikan pernyataan, curhatan, pertanyaan acak, atau membahas topik yang TIDAK ADA HUBUNGANNYA dengan pariwisata, budaya, kuliner, dan informasi kota Palembang, kamu DILARANG KERAS meresponsnya secara natural. Kamu WAJIB membalas dengan kalimat template ini persis: "Maaf, ini adalah aplikasi pariwisata Palembang. Saya kurang mengerti maksud Anda. Ada yang bisa saya bantu seputar wisata Palembang?"

Berikut adalah DATABASE PENGETAHUAN WISATA PALEMBANG:
"""
    try:
        if supabase:
            res = supabase.table("destinations").select("*").execute()
            lines = []
            for d in res.data:
                name = d.get('name', '')
                pmin = d.get('price_min') or 0
                pmax = d.get('price_max') or 0
                if pmin == 0 and pmax == 0:
                    price = "Gratis"
                elif pmin == pmax:
                    price = f"Rp {pmin:,}".replace(",", ".")
                else:
                    price = f"Rp {pmin:,} - Rp {pmax:,}".replace(",", ".")
                
                cat = d.get('category', '')
                hours = d.get('operating_hours') or "-"
                addr = d.get('address') or "-"
                desc = d.get('description_id') or ""
                # hilangkan enter dari deskripsi agar rapi
                desc = desc.replace("\n", " ").strip()
                
                facs = d.get('facilities') or []
                facs_str = ", ".join(facs) if isinstance(facs, list) else str(facs)
                
                lrt = "Ya" if d.get('lrt_accessible') else "Tidak"
                
                info = f"- {name} | Kategori: {cat} | Tiket: {price} | Buka: {hours} | LRT: {lrt} | Fasilitas: {facs_str} | Alamat: {addr} | Deskripsi: {desc}"
                lines.append(info)
            
            base_prompt += "\n".join(lines)
    except Exception as e:
        print(f"Failed to load DB context: {e}")
        
    return base_prompt

class ChatbotModel:
    def __init__(self):
        print("Mempersiapkan model PyTorch/Transformers dari folder ml/saved_models...")
        self.engine = ChatbotEngine()
        self.gemini_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=build_system_prompt()
        )

    def generate_gemini_reply(self, message: str, history: list) -> str:
        global current_key_idx
        MAX_RETRIES = 3
        BASE_WAIT = 5  # Dipercepat karena kita punya kunci cadangan
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                print(f"Fallback to Gemini LLM with context... (attempt {attempt + 1}, using API Key ke-{current_key_idx + 1})")
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
                
                # Hapus karakter markdown ** agar tidak muncul mentah-mentah di aplikasi mobile
                clean_text = response.text.replace("**", "")
                
                return clean_text
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "429" in str(e) or "resource" in error_str or "quota" in error_str or "rate" in error_str
                
                if is_rate_limit:
                    if current_key_idx < len(all_api_keys) - 1:
                        # Swap API Key
                        current_key_idx += 1
                        print(f"API Key ke-{current_key_idx} limit (429). Otomatis swap ke API Key ke-{current_key_idx + 1}...")
                        genai.configure(api_key=all_api_keys[current_key_idx])
                        # Re-instantiate model to ensure it picks up the new config
                        self.gemini_model = genai.GenerativeModel(
                            model_name="gemini-2.5-flash",
                            system_instruction=build_system_prompt()
                        )
                        continue  # Coba lagi tanpa delay karena pakai key baru
                        
                    elif attempt < MAX_RETRIES:
                        # Semua key habis, terpaksa backoff delay
                        wait_time = BASE_WAIT * (2 ** attempt)
                        print(f"Semua API Key limit (429). Retry {attempt + 1}/{MAX_RETRIES} setelah {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                
                print(f"Gemini Fallback Error (final): {e}")
                return "Maaf, saya tidak mengerti maksud Anda. Silakan coba tanyakan hal lain seputar wisata Palembang."

    def generate_reply(self, message: str, history: list = None) -> str:
        if history is None:
            history = []
            
        msg_lower = message.lower()
        
        CONFIDENCE_THRESHOLD = 0.60
        GENERATIVE_INTENTS = {"ask_recommendation", "ask_category", "ask_hidden_gems", "ask_unrelated"}
        ENTITY_DEPENDENT_INTENTS = {"ask_ticket_price", "ask_operating_hours", "ask_destination_info", "ask_lrt_destinations", "ask_location_access", "ask_facilities"}

        # LANGKAH 1 - Jalankan model lokal DULU
        result = self.engine.process_message(message)
        print(f"ML Result: {result}")
        
        intent = result["intent"]
        entities = result["entities"]
        confidence = result.get("confidence", 1.0)

        # Helper function untuk pencarian singkatan manual (Fallback NER)
        def find_destination_by_abbr(text: str):
            text_clean = text.lower().replace("?", "").replace("!", "").replace(".", "").replace(",", "")
            text_padded = f" {text_clean} "
            sorted_abbrs = sorted(ABBREVIATIONS.keys(), key=len, reverse=True)
            for short_name in sorted_abbrs:
                if f" {short_name} " in text_padded:
                    return short_name
            return None
        
        # LANGKAH 2 - MEMORI KONTEKS LOKAL (Pencarian Entity)
        if "DESTINATION" not in entities and intent in ENTITY_DEPENDENT_INTENTS:
            # a. Coba pencocokan singkatan pada pesan saat ini dulu
            found_abbr = find_destination_by_abbr(message)
            if found_abbr:
                entities["DESTINATION"] = found_abbr
                print(f"Memori lokal: Found '{found_abbr}' manually di pesan saat ini.")
            else:
                # b. Telusuri history dari terbaru ke terlama
                print("Memori lokal: Mencari DESTINATION dari riwayat percakapan...")
                for h in reversed(history):
                    if h.get("role") == "user":
                        past_msg = h.get("content", "")
                        # Coba pakai engine untuk masa lalu
                        past_result = self.engine.process_message(past_msg)
                        if "DESTINATION" in past_result["entities"]:
                            entities["DESTINATION"] = past_result["entities"]["DESTINATION"]
                            print(f"Memori lokal: DESTINATION='{entities['DESTINATION']}' dari history (ML) → jawab lokal")
                            break
                        # Coba pakai singkatan untuk masa lalu
                        past_abbr = find_destination_by_abbr(past_msg)
                        if past_abbr:
                            entities["DESTINATION"] = past_abbr
                            print(f"Memori lokal: DESTINATION='{past_abbr}' dari history (Abbr) → jawab lokal")
                            break

        # LANGKAH 3 - GERBANG UTAMA (confidence threshold)
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"Gerbang confidence: {confidence:.2f} < {CONFIDENCE_THRESHOLD:.2f} → Gemini")
            return self.generate_gemini_reply(message, history)

        # Penanganan khusus Hotel (Setelah model yakin, sebelum Langkah 4)
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
            print("Gerbang hotel: Handler rule_hotel dijalankan.")
            return build_response("rule_hotel", {"DAERAH": daerah, "MURAH": murah, "MAHAL": mahal}, message)

        # LANGKAH 4 - Intent generatif (perlu kemampuan LLM meski model yakin)
        if intent in GENERATIVE_INTENTS:
            print(f"Gerbang intent generatif: '{intent}' membutuhkan kreativitas → Gemini")
            return self.generate_gemini_reply(message, history)

        # LANGKAH 5 - Intent terstruktur → jalur LOKAL
        reply_text = build_response(intent, entities, message)
        if "Maaf," in reply_text:
            print("Jalur lokal gagal/Data tidak ditemukan → Gemini")
            return self.generate_gemini_reply(message, history)
            
        print("Jalur lokal berhasil menjawab.")
        return reply_text

# Instansiasi global agar model hanya di-load sekali ke memory
nlp_model = ChatbotModel()

