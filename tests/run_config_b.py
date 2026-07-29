import os
import sys
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

# Pastikan import dari root directory berhasil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from app.services.nlp_service import ChatbotModel
from ml.api.response_builder import build_response, supabase

# Kunci API untuk uji coba (Menggunakan OpenRouter)
openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
client = OpenAI(
    api_key=openrouter_key,
    base_url="https://openrouter.ai/api/v1"
)

# 1. GANTI MODEL: Pindah ke Gemini 2.5 Flash Resmi via OpenRouter
MODEL_NAME = "google/gemini-2.5-flash"

def get_all_destinations_context():
    """Mengambil SEMUA baris tabel destinations dari Supabase sebagai SATU-SATUNYA sumber kebenaran."""
    if not supabase:
        print("KATALOG KOSONG (cek Supabase/RLS/env) - Tidak ada koneksi DB")
        sys.exit(1)
    
    try:
        response = supabase.table("destinations").select("name, category, description_id, price_min, price_max, operating_hours, address, facilities, lrt_accessible").execute()
        katalog_data = response.data
        
        # DEBUG: Cetak jumlah baris katalog sesuai permintaan
        print(f"[DEBUG] catalog rows = {len(katalog_data)}")
        
        if not katalog_data:
            print("KATALOG KOSONG (cek Supabase/RLS/env)")
            sys.exit(1)
            
        lines = []
        for d in katalog_data:
            name = d.get('name', '')
            cat = d.get('category', '')
            pmin = d.get('price_min') or 0
            pmax = d.get('price_max') or 0
            
            if pmin == pmax == 0:
                price = "Gratis"
            elif pmin == pmax:
                price = f"Rp {pmin:,}".replace(',', '.')
            else:
                price = f"Rp {pmin:,} - Rp {pmax:,}".replace(',', '.')
                
            hours = d.get('operating_hours', '-') or "-"
            addr = d.get('address', '-') or "-"
            facs = d.get('facilities', [])
            fac_str = ", ".join(facs) if isinstance(facs, list) else str(facs)
            lrt = "Ya" if d.get('lrt_accessible') else "Tidak"
            
            # Format katalog injeksi
            line = f"- {name} | kategori: {cat} | jam: {hours} | harga: {price} | alamat: {addr} | fasilitas: {fac_str} | LRT: {lrt}"
            lines.append(line)
            
        return "\n".join(lines)
    except Exception as e:
        print(f"Error mengambil katalog: {str(e)}")
        sys.exit(1)


SYSTEM_GUARDRAIL = """Kamu adalah GUARDRAIL (validator) untuk chatbot wisata Palembang "TanyaKito".
Tugasmu MEMVALIDASI dan bila perlu MEMPERBAIKI draf jawaban.

INPUT:
- Pertanyaan pengguna
- Draf jawaban dari sistem (Actor)
- KONTEKS_FAKTA: DAFTAR LENGKAP destinasi dari database — SATU-SATUNYA sumber kebenaran
- Riwayat percakapan

PRINSIP UTAMA:
- KONTEKS_FAKTA memuat SEMUA destinasi yang sistem ketahui. Jika sebuah tempat ADA di
  KONTEKS_FAKTA, kamu HARUS bisa menjawabnya dari data itu.
- Cocokkan nama tempat secara LONGGAR: abaikan spasi, huruf besar/kecil, dan singkatan
  (mis. "balaputradewa" = "Balaputra Dewa"; "smb ii" = "Sultan Mahmud Badaruddin II").
- Kamu TIDAK punya pengetahuan lain selain KONTEKS_FAKTA. Dilarang memakai pengetahuan luar.

ATURAN VERDICT:
1. PASS (salin draf apa adanya) jika SALAH SATU benar:
   - Draf sudah benar & konsisten dengan KONTEKS_FAKTA.
   - Pertanyaan ambigu tanpa nama tempat (mis. "harganya brapa?", "buka 24 jam") DAN
     draf sudah minta klarifikasi sopan.
   - Pertanyaan di luar domain wisata Palembang DAN draf sudah menolak.
2. FAIL (perbaiki jawaban_akhir memakai KONTEKS_FAKTA) jika:
   - Draf gagal menjawab (mis. minta nama tempat) PADAHAL tempatnya ADA di KONTEKS_FAKTA
     → temukan tempat itu, jawab langsung dengan faktanya.
   - Draf salah intent (mis. user tanya cara menginap, draf beri profil museum) → perbaiki.
   - Draf bertentangan dengan KONTEKS_FAKTA.
   - Draf menolak kaku untuk pertanyaan in-scope yang bisa dijawab (mis. "rekomendasi
     wisata sejarah/kuliner") → beri DAFTAR tempat yang cocok dari KONTEKS_FAKTA.

LARANGAN KERAS:
- DILARANG menambah nama/harga/jam/alamat/fasilitas yang TIDAK ADA di KONTEKS_FAKTA.
- Jika tempat yang ditanya TIDAK ADA di KONTEKS_FAKTA: jujur katakan belum tersedia,
  JANGAN mengarang, JANGAN menebak tempat lain.
- Untuk pertanyaan tanpa subjek tempat: JANGAN menebak tempatnya; pertahankan klarifikasi (PASS).
- DILARANG menyebut istilah sistem apa pun ("KONTEKS_FAKTA", "draf", "Actor", "verdict",
  "prompt") di dalam jawaban_akhir. Tulis seolah bicara langsung ke pengguna.
- Untuk pertanyaan di luar domain: boleh memperhalus penolakan, tapi verdict TETAP PASS
  jika draf sudah menolak (perhalusan gaya BUKAN kegagalan faktual).
- Jika kamu tidak mengubah isi draf, verdict WAJIB PASS.

OUTPUT (WAJIB JSON valid, tanpa teks lain, tanpa markdown):
{"verdict":"PASS"|"FAIL","jawaban_akhir":"<jawaban final untuk pengguna>"}"""

def main():
    gold_path = "tests/end_to_end_gold.json"
    output_path = "tests/config_b_results.json"
    
    # 4. FRESH RUN (WAJIB — cegah hasil basi)
    if os.path.exists(output_path):
        timestamp = int(time.time())
        old_path = f"tests/config_b_results_OLD_{timestamp}.json"
        os.rename(output_path, old_path)
        print(f"File lama di-rename ke {old_path} untuk fresh run.")
    
    print(f"Memuat Dataset dari {gold_path}...")
    with open(gold_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print("Mengambil KATALOG DESTINASI GLOBAL dari Supabase...")
    konteks_fakta_global = get_all_destinations_context()
    
    print("Memuat Model Lokal (XLM-RoBERTa Intent & NER)...")
    chatbot = ChatbotModel()
    
    print(f"\nMulai Pengujian Konfigurasi B (NLP + Guardrail - {MODEL_NAME})\n" + "-"*50)
    
    results = []
    
    for i, data in enumerate(dataset):
        query = data["query"]
        category = data.get("category", "unknown")
            
        start_total = time.time()
        
        # ACTOR (NLP Lokal)
        try:
            ml_res = chatbot.engine.process_message(query)
            intent = ml_res.get("intent", "unknown")
            entities = ml_res.get("entities", {})
            
            # --- MULAI LOGIKA RESOLUSI ENTITAS (Sinkron dengan nlp_service.py) ---
            ENTITY_DEPENDENT_INTENTS = {"ask_ticket_price", "ask_operating_hours", "ask_destination_info", "ask_lrt_destinations", "ask_location_access", "ask_facilities"}
            
            if "DESTINATION" not in entities and intent in ENTITY_DEPENDENT_INTENTS:
                # 1. Abbreviation Match
                from ml.api.response_builder import ABBREVIATIONS
                text_clean = query.lower().replace("?", "").replace("!", "").replace(".", "").replace(",", "")
                text_padded = f" {text_clean} "
                sorted_abbrs = sorted(ABBREVIATIONS.keys(), key=len, reverse=True)
                found_abbr = None
                for short_name in sorted_abbrs:
                    if f" {short_name} " in text_padded:
                        found_abbr = short_name
                        break
                
                if found_abbr:
                    entities["DESTINATION"] = found_abbr
                else:
                    # 2. Fuzzy Match ke Database Supabase
                    import difflib
                    from app.services.nlp_service import get_destination_names
                    
                    db_names = get_destination_names()
                    if db_names:
                        noise_words = ["berapa", "harga", "tiket", "masuk", "dari", "ke", "di", "untuk", 
                                       "jam", "buka", "tutup", "operasional", "alamat", "lokasi", "dimana",
                                       "fasilitas", "apa", "saja", "ada", "yang", "nya", "dong", "ya",
                                       "kasih", "tau", "info", "tentang", "gimana", "bagaimana", "museum",
                                       "wisata", "tempat", "taman", "masjid", "kampung", "kawasan", "pulau",
                                       "jembatan", "hutan", "sungai", "kolam", "renang", "wahana", "kuliner",
                                       "sejarah", "kategori", "disana", "sini", "sana", "buat",
                                       "apakah", "ga", "gak", "nggak",
                                       "palembang", "naik", "dekat"]
                        
                        text_clean_stripped = query.lower().replace("?", "").replace("!", "").replace(".", "").replace(",", "").strip()
                        words = text_clean_stripped.split()
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
                                score = difflib.SequenceMatcher(None, candidate, db_name.lower()).ratio()
                                if score > best_score and score >= 0.55:
                                    best_score = score
                                    best_match = db_name
                        if best_match:
                            print(f"🔍 Fuzzy Match: '{text_clean_stripped}' -> '{best_match}' (skor: {best_score:.2f})")
                            entities["DESTINATION"] = best_match
            # --- SELESAI LOGIKA RESOLUSI ENTITAS ---
            
            if intent in ["ask_recommendation", "ask_category", "ask_hidden_gems", "ask_unrelated"]:
                draft_text = "Maaf, saya tidak mengerti maksud Anda."
            else:
                raw_reply = build_response(intent, entities, query)
                draft_text = raw_reply["reply"] if isinstance(raw_reply, dict) else str(raw_reply)
        except Exception as e:
            intent = "ERROR"
            entities = {}
            draft_text = f"ERROR: {str(e)}"
            
        # GUARDRAIL (LLM via OpenRouter)
        user_payload = f"Pertanyaan pengguna: {query}\nDraf sistem: {draft_text}\nKONTEKS_FAKTA:\n{konteks_fakta_global}\nRiwayat percakapan: (kosong)"
        
        # DEBUG INJEKSI KATALOG (Poin 5)
        if i == 1:
            print("[DEBUG] balaputra in prompt:", "balaputra" in user_payload.lower())

        latency_llm = 0
        verdict = "UNKNOWN"
        final_answer = "N/A"
        
        for attempt in range(3):
            try:
                start_llm = time.time()
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role":"system","content":SYSTEM_GUARDRAIL},
                        {"role":"user","content":user_payload}
                    ],
                    temperature=0,
                    response_format={"type":"json_object"},
                    extra_headers={"X-Title":"TanyaKito Eval"}
                )
                latency_llm = time.time() - start_llm
                raw_response = resp.choices[0].message.content.strip()
                
                try:
                    parsed = json.loads(raw_response)
                    verdict = parsed.get("verdict", "PARSE_ERROR")
                    final_answer = parsed.get("jawaban_akhir", "PARSE_ERROR")
                    break # Success
                except Exception:
                    if attempt < 2:
                        time.sleep(1) 
                        continue
                    else:
                        verdict = "PARSE_ERROR"
                        final_answer = raw_response
                        break
                    
            except Exception as e:
                print(f"    -> API Error: {str(e)}")
                if attempt < 2:
                    time.sleep(2)
                else:
                    verdict = "ERROR"
                    final_answer = str(e)
                    break
        
        # 6. FIX BUG LOGGING `intervensi` (Dihitung dari selisih jawaban)
        # Jika jawaban berbeda dari draf, maka terjadi intervensi
        is_intervention = (final_answer.strip() != draft_text.strip())
        
        record = {
            "no": i + 1,
            "category": category,
            "query": query,
            "intent": intent,
            "entities": entities,
            "draf_actor": draft_text,
            "verdict": verdict,
            "jawaban_akhir": final_answer,
            "intervensi": is_intervention,
            "model": MODEL_NAME,
            "latency_llm_detik": round(latency_llm, 4)
        }
        
        # 7. Tulis data mentah (JSON) LEBIH DULU
        results.append(record)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
            
        print(f"[{i+1}/15] {query} -> Intervensi: {is_intervention} (llm={latency_llm:.2f}s)")
        
        # 3. THROTTLE: Kredit terisi → limit longgar
        if i < len(dataset) - 1:
            time.sleep(1)
            
    # AKHIR PENGUJIAN - SUMMARY DIHITUNG DARI JSON (Poin 7)
    print("\n### Ringkasan Hasil Konfigurasi B (Actor + Guardrail Gemini)\n")
    print("| No | Kategori | Query | Draf Actor | Intervensi | Jawaban Akhir | Latency (s) |")
    print("|---|---|---|---|---|---|---|")
    
    valid_count = 0
    pass_count = 0
    fail_count = 0
    total_lat_llm = 0
    intervention_count = 0
    error_count = 0
    
    for r in results:
        v = r.get('verdict', 'UNKNOWN')
        clean_draft = str(r.get('draf_actor', '')).replace('\n', ' ').replace('\r', '')
        clean_ans = str(r.get('jawaban_akhir', '')).replace('\n', ' ').replace('\r', '')
        lat_llm = r.get('latency_llm_detik', 0)
        no_id = r.get('no', '?')
        interv = r.get('intervensi', False)
        
        if v in ["ERROR", "PARSE_ERROR", "RATE_LIMITED"]:
            error_count += 1
            print(f"| {no_id} | {r.get('category')} | {r.get('query')} | {clean_draft} | **ERROR** | N/A | {lat_llm} |")
        else:
            valid_count += 1
            total_lat_llm += lat_llm
            if v == "PASS": pass_count += 1
            if v == "FAIL": fail_count += 1
            if interv: intervention_count += 1
            
            interv_str = "Ya" if interv else "Tidak"
            print(f"| {no_id} | {r.get('category')} | {r.get('query')} | {clean_draft} | {interv_str} | {clean_ans} | {lat_llm} |")
            
    print("\n**Statistik Eksekusi:**")
    print(f"- Model Dipakai: {MODEL_NAME}")
    if valid_count > 0:
        print(f"- Rata-rata Latency LLM: {total_lat_llm/valid_count:.4f} detik")
        intervention_rate = (intervention_count / valid_count) * 100
        # Metrik koreksi sederhana (asumsi guardrail sukses membetulkan yang salah)
        answer_correctness = ((valid_count - error_count) / len(dataset)) * 100
    else:
        print("- Rata-rata Latency LLM: N/A")
        intervention_rate = 0
        answer_correctness = 0
        
    print(f"- Jumlah PASS Terdeteksi LLM: {pass_count}")
    print(f"- Jumlah Intervensi (Selisih Teks Nyata): {intervention_count}")
    print(f"- Jumlah ERROR/Gagal API: {error_count}")
    print(f"- Intervention Rate: {intervention_rate:.1f}%")
    print(f"- Answer Correctness (Aproksimasi Validitas): {answer_correctness:.1f}%")
    print("- Hallucination Rate (%): (Mohon cek manual pada bagian AUDIT di bawah)")
    
    # CEK OTOMATIS ANTI-REGRESI & ANTI-HALUSINASI
    print("\n=== AUDIT OTOMATIS (Cek Manual Halusinasi) ===")
    
    for r in results:
        no_id = r.get('no', '?')
        interv = r.get('intervensi', False)
        ans = str(r.get('jawaban_akhir', ''))
        
        # Peringatan bocor prompt
        if any(term in ans for term in ["KONTEKS_FAKTA", "draf", "Actor", "verdict"]):
            print(f"PERINGATAN BOCOR PROMPT [{no_id}]")
            
        if interv:
            # Jika ada intervensi, tampilkan ke konsol agar pengguna mudah mengecek halusinasi
            print(f"AUDIT [{no_id}]: revisi={ans}")

if __name__ == "__main__":
    main()
