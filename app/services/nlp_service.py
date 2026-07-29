import os
import time
from ml.api.engine import ChatbotEngine
from ml.api.response_builder import build_response, ABBREVIATIONS, supabase

# Konfigurasi OpenRouter (menggunakan SDK OpenAI)
openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

from openai import OpenAI
llm_client = OpenAI(
    api_key=openrouter_key,
    base_url="https://openrouter.ai/api/v1"
)
current_key_idx = 0
free_keys = [openrouter_key]  # Pertahankan variabel ini agar tidak merusak logika fallback error 429
import difflib

# Cache global untuk nama destinasi (mencegah query berulang ke Supabase setiap panggil fuzzy)
DESTINATIONS_CACHE = []

def get_destination_names():
    global DESTINATIONS_CACHE
    if not DESTINATIONS_CACHE and supabase:
        res = supabase.table("destinations").select("name").execute()
        if res.data:
            DESTINATIONS_CACHE = [d["name"] for d in res.data]
    return DESTINATIONS_CACHE


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
        self.llm_model = "google/gemini-2.5-flash"

    def evaluate_with_guardrail(self, message: str, draft_reply: str, history: list) -> dict:
        global current_key_idx, llm_client, free_keys
        MAX_RETRIES = 3
        BASE_WAIT = 5
        
        history_text = "\n".join([f"{h.get('role', 'user').capitalize()}: {h.get('content', '')}" for h in history[-3:]]) if history else "Tidak ada riwayat."
        
        db_context = build_system_prompt()
        
        guardrail_prompt = f"""Anda adalah AI Guardrail (Juri Penilai) untuk Chatbot Pelesir Palembang.
Tugas Anda adalah mengevaluasi Draf Balasan dari NLP Lokal terhadap pesan pengguna.

[REFERENSI DATABASE]
{db_context}

[KONTEKS OBROLAN]
Riwayat Obrolan:
{history_text}

Pesan Pengguna Saat Ini: "{message}"
Draf Balasan Lokal: "{draft_reply}"

ATURAN KETAT:
1. Jika Draf Balasan akurat, informatif, dan menyambung dengan konteks (ATAU jika draf dengan cerdas meminta klarifikasi seperti "Boleh sebutkan nama lokasinya?"), Anda HARUS membalas tepat dengan 1 kata: PASS
2. Jika Draf Balasan ngawur, salah sasaran, hanya mengatakan "Maaf, saya tidak mengerti", atau sama sekali tidak informatif, Anda HARUS menolaknya dengan membalas format: FAIL: <tuliskan_jawaban_baru_yang_benar_disini_berdasarkan_REFERENSI_DATABASE>
3. Palembang BUKAN kota pesisir laut. Jika ditanya soal laut/pantai alami, arahkan ke Sungai Musi atau Pulau Kemaro.

Evaluasi Anda:"""

        for attempt in range(MAX_RETRIES + 1):
            try:
                print(f"🛡️ Guardrail mengecek Draf... (attempt {attempt + 1})")
                print(f"   [Draf Lokal]: {draft_reply}")
                
                chat_completion = llm_client.chat.completions.create(
                    messages=[{"role": "user", "content": guardrail_prompt}],
                    model=self.llm_model,
                    temperature=0.1,  # Suhu rendah agar stabil membalas PASS
                )
                response_text = chat_completion.choices[0].message.content.strip()
                
                if response_text.upper().startswith("PASS"):
                    print("✅ Guardrail: PASS (Meneruskan jawaban lokal)")
                    return {"reply": draft_reply, "source": "lokal"}
                else:
                    print(f"❌ Guardrail: FAIL (LLM mengambil alih)")
                    print(f"   [Alasan/Jawaban LLM]: {response_text}")
                    # Ambil teks setelah FAIL:
                    clean_text = response_text.replace("FAIL:", "", 1).strip()
                    clean_text = clean_text.replace("**", "")
                    
                    from ml.api.response_builder import enrich_gemini_response
                    rich_result = enrich_gemini_response(clean_text)
                    rich_result["source"] = "gemini_guardrail"
                    return rich_result
                
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "429" in str(e) or "rate" in error_str or "limit" in error_str
                
                if is_rate_limit:
                    if current_key_idx < len(free_keys) - 1:
                        current_key_idx += 1
                        print(f"API Key ke-{current_key_idx} limit. Swap ke Key ke-{current_key_idx + 1}...")
                        llm_client = OpenAI(
                            api_key=free_keys[current_key_idx],
                            base_url="https://openrouter.ai/api/v1"
                        )
                        continue
                        
                    elif attempt < MAX_RETRIES:
                        wait_time = BASE_WAIT * (2 ** attempt)
                        print(f"Semua Key limit. Retry {attempt + 1}/{MAX_RETRIES} setelah {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                
                print(f"⚠️ Guardrail Error: {e} -> Bypass ke lokal")
                # Jika Groq mati / timeout, biarkan jawaban lokal lewat sebagai fallback terakhir
                return {"reply": draft_reply, "source": "lokal_bypass"}


    def generate_reply(self, message: str, history: list = None) -> str:
        if history is None:
            history = []
            
        msg_lower = message.lower()
        
        GENERATIVE_INTENTS = {"ask_recommendation", "ask_category", "ask_hidden_gems", "ask_unrelated"}
        ENTITY_DEPENDENT_INTENTS = {"ask_ticket_price", "ask_operating_hours", "ask_destination_info", "ask_lrt_destinations", "ask_location_access", "ask_facilities"}

        # LANGKAH 1 - Jalankan model lokal DULU
        result = self.engine.process_message(message)
        print(f"ML Result: {result}")
        
        intent = result["intent"]
        entities = result["entities"]
        confidence = result.get("confidence", 1.0)
        
        if confidence < 0.65:
            print(f"⚠️ LOW CONFIDENCE INTENT: {intent} ({confidence:.2f}). Fallback ke ask_unrelated.")
            intent = "ask_unrelated"

        # INTERCEPT ROUTING EKSPLISIT (Origin -> Destination)
        import re
        
        origin_str, dest_str = None, None
        
        # 1. rute dari A ke B / dari A ke B gimana
        route_match = re.search(r'(?:rute|jalan|arah|panduan|cara).*?(?:dari|dri|dr)\s+(.+?)\s+(?:ke|k)\s+(.+)', msg_lower)
        if not route_match:
            route_match = re.search(r'(?:dari|dri|dr)\s+(.+?)\s+(?:ke|k)\s+(.+?)(?:\s+gimana|\s+gmna|\s+bagaimana|\s+rutenya|\s+caranya|\?|$)', msg_lower)
        if not route_match:
            # 2. kalau A ke B / kalau dari A ke B
            route_match = re.search(r'^(?:kalau|kalo|klo)\s+(?:(?:dari|dri|dr)\s+)?(?!pergi|mau|ingin|jalan|liburan)(.+?)\s+(?:ke|k)\s+(.+?)(?:\s+gimana|\s+bagaimana|\s+rutenya|\s+caranya|\?|$)', msg_lower)
            
        if route_match:
            origin_str, dest_str = route_match.group(1), route_match.group(2)
        else:
            # 3. cara ke B dari A / ke B naik apa dari A
            route_match_2 = re.search(r'(?:cara|naik|rute).*?(?:ke|k)\s+(.+?)\s+.*?(?:dari|dri|dr)\s+(.+)', msg_lower)
            if not route_match_2:
                # 4. ke B dari A (simple)
                route_match_2 = re.search(r'(?:ke|k)\s+(.+?)\s+(?:bsa|bisa|naik|lwat|lewat|dari|dri|dr).*?(?:dari|dri|dr)\s+(.+)', msg_lower)
            if route_match_2:
                dest_str, origin_str = route_match_2.group(1), route_match_2.group(2)
                
        if origin_str and dest_str:
            # Bersihkan tanda baca di akhir dan kata-kata noise
            noise_words = ['gmna', 'gimana', 'y', 'ya', 'sih', 'dong', 'bang', 'lwt mn', 'lwt man', 'lewat mana', 'lewat mn']
            for noise in noise_words:
                origin_str = re.sub(rf'\b{noise}\b', '', origin_str).strip()
                dest_str = re.sub(rf'\b{noise}\b', '', dest_str).strip()
            
            origin_str = origin_str.replace('?','').replace('.','').strip()
            dest_str = dest_str.replace('?','').replace('.','').strip()
            intent = "ask_route"
            entities = {"ORIGIN": origin_str, "DESTINATION": dest_str}
            confidence = 1.0
            print(f"Memori lokal: Intercepted route query! Origin: {origin_str}, Dest: {dest_str}")

        # Helper function untuk pencarian singkatan manual (Fallback NER)
        def find_destination_by_abbr(text: str):
            text_clean = text.lower().replace("?", "").replace("!", "").replace(".", "").replace(",", "")
            text_padded = f" {text_clean} "
            sorted_abbrs = sorted(ABBREVIATIONS.keys(), key=len, reverse=True)
            for short_name in sorted_abbrs:
                if f" {short_name} " in text_padded:
                    return short_name
            return None

        # Konstanta Threshold untuk Fuzzy Match
        FUZZY_MATCH_THRESHOLD = 0.55  # Pertahankan 0.55 agar kasus seperti Punti Kayu tetap lolos

        # Helper function untuk fuzzy matching ke database Supabase
        def find_destination_fuzzy(text: str, threshold: float = FUZZY_MATCH_THRESHOLD):
            """Cocokkan sisa teks query ke daftar nama destinasi di database menggunakan fuzzy matching."""
            try:
                db_names = get_destination_names()
                if not db_names:
                    return None
                
                # Bersihkan teks query dari noise
                text_clean = text.lower().replace("?", "").replace("!", "").replace(".", "").replace(",", "").strip()
                # Hapus kata-kata umum non-destinasi agar matching lebih akurat
                noise_words = ["berapa", "harga", "tiket", "masuk", "dari", "ke", "di", "untuk", 
                               "jam", "buka", "tutup", "operasional", "alamat", "lokasi", "dimana",
                               "fasilitas", "apa", "saja", "ada", "yang", "nya", "dong", "ya",
                               "kasih", "tau", "info", "tentang", "gimana", "bagaimana", "museum",
                               "wisata", "tempat", "taman", "masjid", "kampung", "kawasan", "pulau",
                               "jembatan", "hutan", "sungai", "kolam", "renang", "wahana", "kuliner",
                               "sejarah", "kategori", "disana", "sini", "sana", "buat",
                               "apakah", "ga", "gak", "nggak",
                               "palembang", "naik", "dekat"]
                words = text_clean.split()
                # Bangun kandidat: coba semua substring 1-4 kata berturut-turut
                candidates = []
                for length in range(len(words), 0, -1):
                    for start in range(len(words) - length + 1):
                        chunk = " ".join(words[start:start+length])
                        chunk_words = chunk.split()
                        
                        # Hapus chunk yang HANYA berisi noise words
                        if all(w in noise_words for w in chunk_words):
                            continue
                            
                        # SKIP jika chunk yang tersisa (setelah dibersihkan) sangat pendek (< 4 karakter)
                        clean_chunk_words = [w for w in chunk_words if w not in noise_words]
                        clean_chunk = " ".join(clean_chunk_words).strip()
                        if len(clean_chunk) < 4:
                            continue
                            
                        candidates.append(chunk)
                
                best_match = None
                best_score = 0.0
                
                for candidate in candidates:
                    for db_name in db_names:
                        # Bandingkan candidate dengan nama db (case-insensitive)
                        score = difflib.SequenceMatcher(None, candidate, db_name.lower()).ratio()
                        if score > best_score and score >= threshold:
                            best_score = score
                            best_match = db_name
                            
                if best_match:
                    print(f"🔍 Fuzzy Match: '{text_clean}' -> '{best_match}' (skor: {best_score:.2f})")
                    return best_match
            except Exception as e:
                print(f"Fuzzy match error: {e}")
            return None

        # LANGKAH 1.5 - MEMORI INTENT (Follow-up context)
        is_follow_up = any(msg_lower.startswith(w) for w in ["kalo ", "kalau ", "gimana ", "bagaimana "])
        
        # HANYA warisi intent masa lalu JIKA intent saat ini tidak eksplisit spesifik
        if is_follow_up and history and (intent not in ENTITY_DEPENDENT_INTENTS or intent == "ask_destination_info"):
            print("Memori lokal: Pesan ambigu/follow-up. Mencari intent sebelumnya...")
            for h in reversed(history):
                if h.get("role") == "user":
                    past_msg = h.get("content", "")
                    past_result = self.engine.process_message(past_msg)
                    past_intent = past_result["intent"]
                    if past_intent in ENTITY_DEPENDENT_INTENTS:
                        intent = past_intent
                        print(f"Memori lokal: Mewarisi intent '{intent}' dari histori.")
                        
                        # Jika model gagal mendeteksi DESTINATION di pesan pendek ini, paksa sisa kalimat sebagai DESTINATION
                        if "DESTINATION" not in entities:
                            sisa_kata = msg_lower
                            for w in ["kalo di ", "kalau di ", "kalo ", "kalau ", "gimana dengan ", "bagaimana dengan ", "gimana ", "bagaimana "]:
                                if sisa_kata.startswith(w):
                                    sisa_kata = sisa_kata.replace(w, "", 1).strip()
                                    break
                            if sisa_kata:
                                entities["DESTINATION"] = sisa_kata
                                print(f"Memori lokal: Memaksa sisa kalimat '{sisa_kata}' sebagai DESTINATION.")
                        break

        # LANGKAH 1.8 - FUZZY MATCHING KE DATABASE (Sebelum Fallback Histori)
        # Urutan resolusi: NER (Langkah 1) → Abbreviation → Fuzzy DB Match → Histori (Langkah 2)
        # Ini mencegah sistem salah mengambil destinasi dari histori padahal nama destinasi
        # masih ada di dalam kalimat query saat ini (hanya gagal ditangkap oleh NER).
        if "DESTINATION" not in entities and intent in ENTITY_DEPENDENT_INTENTS:
            # a. Coba pencocokan singkatan pada pesan saat ini dulu
            found_abbr = find_destination_by_abbr(message)
            if found_abbr:
                entities["DESTINATION"] = found_abbr
                print(f"Memori lokal: Found '{found_abbr}' manually di pesan saat ini.")
            else:
                # b. (BARU!) Coba fuzzy matching ke database Supabase
                fuzzy_result = find_destination_fuzzy(message)
                if fuzzy_result:
                    entities["DESTINATION"] = fuzzy_result
                    print(f"🔍 Memori lokal: DESTINATION='{fuzzy_result}' dari Fuzzy DB Match → jawab lokal")
                else:
                    # c. Telusuri history dari terbaru ke terlama (opsi TERAKHIR)
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
                            
        # LANGKAH 2.5 - JEBAKAN INTENT (Kesalahan Klasifikasi akibat Kata Kunci)
        # Kasus: User meminta rekomendasi wisata berdasar atribut ("wisata yang buka 24 jam")
        # tapi NLP malah mendeteksi `ask_operating_hours` padahal tidak ada entitas destinasi.
        is_asking_recommendation = any(kw in msg_lower for kw in ["berikan", "rekomendasi", "apa saja", "wisata yang", "tempat yang", "kasih tau wisata", "cari wisata", "ada gak wisata", "ada nggak wisata", "carikan", "kmn", "kemana", "ke mana", "bgus ny", "bagus nya", "dimana aja", "di mana aja", "dmn aj"])
        if intent in ENTITY_DEPENDENT_INTENTS and "DESTINATION" not in entities and is_asking_recommendation:
            print(f"Jebakan Intent: Deteksi kata pencarian/rekomendasi. Mengubah intent dari '{intent}' menjadi 'ask_recommendation' → Gemini")
            intent = "ask_recommendation"

        # LANGKAH 3 - PRODUKSI DRAF LOKAL (The Actor)
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
            draft_reply = build_response("rule_hotel", {"DAERAH": daerah, "MURAH": murah, "MAHAL": mahal}, message)
        else:
            draft_reply = build_response(intent, entities, message)

        # LANGKAH 4 - EVALUASI OLEH JURI (The Critic / Guardrail)
        final_verdict = self.evaluate_with_guardrail(message, draft_reply, history)
        
        # LANGKAH 5 - BUNGKUS DENGAN RICH RESPONSES
        if final_verdict["source"] == "lokal":
            # Jika lolos sensor Juri, bungkus dengan fitur lengkap peta & kartu
            from ml.api.response_builder import build_rich_response
            rich_response = build_rich_response(intent, entities, final_verdict["reply"])
            rich_response["source"] = "lokal"
            return rich_response
        else:
            # Jika Juri membatalkan dan merangkai baru, kembalikan hasil Juri (yang sudah di-enrich)
            return final_verdict

# Instansiasi global agar model hanya di-load sekali ke memory
nlp_model = ChatbotModel()

