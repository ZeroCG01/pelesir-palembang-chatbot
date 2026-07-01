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
        
        # --- RULE-BASED INTERCEPTOR ---
        # 1. Intercept Itinerary (SERAHKAN KE GEMINI AGAR DINAMIS)
        if "itinerary" in msg_lower or "jadwal" in msg_lower or ("hari" in msg_lower and "wisata" in msg_lower):
            return self.generate_gemini_reply(message, history)
            
        # 1.5 Intercept Multi-Intent (Pertanyaan Ganda)
        has_price = "harga" in msg_lower or "tiket" in msg_lower or "biaya" in msg_lower
        has_time = "jam" in msg_lower or "buka" in msg_lower or "tutup" in msg_lower
        has_location = "dimana" in msg_lower or "lokasi" in msg_lower or "alamat" in msg_lower
        
        if (has_price and has_time) or (has_price and has_location) or (has_time and has_location):
            return self.generate_gemini_reply(message, history)

        # 1.6 Intercept Permintaan Detail
        if "detail" in msg_lower or "lengkap" in msg_lower:
            return self.generate_gemini_reply(message, history)

        # 1.7 Intercept Rekomendasi / Hidden Gems / Kategori (SERAHKAN KE GEMINI)
        recommendation_words = ["rekomendasi", "rekomendasikan", "recommend", "hidden gems", "hidden gem", "saran", "sarankan"]
        if any(rw in msg_lower for rw in recommendation_words):
            return self.generate_gemini_reply(message, history)

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
        # Coba cocokkan nama destinasi jika ML NER tidak mendeteksi
        if "DESTINATION" not in entities:
            # Hapus tanda baca umum agar "bkb?" menjadi "bkb"
            msg_clean_punct = message.lower().replace("?", "").replace("!", "").replace(".", "").replace(",", "")
            msg_lower_padded = f" {msg_clean_punct} "
            
            # Sort by longest key first agar multi-word match duluan (misal "kampung kapitan" sebelum "ki")
            sorted_abbrs = sorted(ABBREVIATIONS.keys(), key=len, reverse=True)
            for short_name in sorted_abbrs:
                if f" {short_name} " in msg_lower_padded:
                    entities["DESTINATION"] = short_name
                    print(f"Fallback NER: Found '{short_name}' manually (after removing punctuation).")
                    break
        
        # 2. Hasilkan balasan natural (melibatkan database Supabase di dalam builder)
        reply_text = build_response(intent, entities)
        
        # 3. TRIGGER GEMINI FALLBACK (Smarter Heuristics)
        fallback_reasons = []
        msg_clean = message.lower().strip()
        
        # 3a. Jika ML lokal menjawab dengan error/template gagal
        if "Maaf," in reply_text:
            fallback_reasons.append("Pesan error default lokal")
        
        # 3b. Intent yang sebaiknya selalu ditangani Gemini
        if intent in ["ask_unrelated", "ask_category", "ask_recommendation", "ask_hidden_gems"]:
            fallback_reasons.append("Pertanyaan rekomendasi, hidden gems, kategori, atau out-of-domain diserahkan ke Gemini")
                
        # 3c. Pertanyaan lanjutan (follow-up) berdasarkan kata kunci awalan
        follow_up_words = ["kalau ", "kalo ", "bagaimana dengan ", "gimana dengan ", "gimana kalo ", "gimana ", "gmana ", "lalu ", "terus ", "trus ", "alam", "sejarah", "kuliner", "religi", "budaya", "taman"]
        if any(msg_clean.startswith(w) for w in follow_up_words):
            fallback_reasons.append("Pertanyaan lanjutan (membutuhkan history)")
            
        # 3d. Pertanyaan pendek tanpa kata tanya (indikasi follow-up)
        words = msg_clean.split()
        if len(words) <= 5 and "DESTINATION" in entities:
            question_words = ["apa", "info", "deskripsi", "dimana", "di mana", "lokasi", "berapa", "harga", "tiket", "jam", "buka", "tutup"]
            if not any(q in msg_clean for q in question_words):
                fallback_reasons.append("Kalimat pendek tanpa kata tanya (indikasi follow-up)")

        # 3e. Pertanyaan tanpa entity TAPI ada history (kemungkinan follow-up kontekstual)
        if "DESTINATION" not in entities and len(history) > 0:
            # Jika intent butuh entity tapi tidak ada, kemungkinan besar user lanjut dari percakapan sebelumnya
            entity_dependent_intents = ["ask_ticket_price", "ask_operating_hours", "ask_destination_info", "ask_lrt_destinations", "ask_location_access", "ask_facilities"]
            if intent in entity_dependent_intents:
                fallback_reasons.append("Intent butuh entity tapi tidak ada — kemungkinan follow-up kontekstual")

        # 3f. Pertanyaan perbandingan atau opini (selalu serahkan ke Gemini)
        comparison_words = ["lebih bagus", "lebih baik", "dibanding", "banding", "versus", "vs ", "atau ", "pilih mana", "mending"]
        if any(cw in msg_clean for cw in comparison_words):
            fallback_reasons.append("Pertanyaan perbandingan/opini (serahkan ke Gemini)")
        
        # 3g. Pertanyaan dengan kata tanya umum tanpa kata kunci spesifik intent
        generic_question_words = ["kenapa", "mengapa", "bagaimana", "gimana", "gmana", "apakah", "bisakah", "bolehkah"]
        if any(msg_clean.startswith(gq) for gq in generic_question_words) and "DESTINATION" not in entities:
            fallback_reasons.append("Pertanyaan generik tanpa entity (serahkan ke Gemini)")

        # 3h. Confidence Score Rendah (Model ML Ragu-ragu)
        confidence = result.get("confidence", 1.0)
        if confidence < 0.80:
            fallback_reasons.append(f"Confidence ML lokal rendah ({confidence:.2f} < 0.80)")

        if len(fallback_reasons) > 0:
            print(f"Trigger Gemini Fallback karena: {fallback_reasons}")
            return self.generate_gemini_reply(message, history)
        
        # Jika bukan error dan bukan follow-up, kembalikan jawaban dari ML Lokal
        return reply_text

# Instansiasi global agar model hanya di-load sekali ke memory
nlp_model = ChatbotModel()

