import os
import sys
import json
import time
from dotenv import load_dotenv

# Pastikan import dari root directory berhasil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from app.services.nlp_service import ChatbotModel
from ml.api.response_builder import build_response

def main():
    print("Memuat Dataset dari tests/end_to_end_gold.json...")
    with open("tests/end_to_end_gold.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print("Memuat Model Lokal (XLM-RoBERTa Intent & NER)...")
    chatbot = ChatbotModel()
    
    results = []
    total_latency = 0
    
    print("\nMulai Pengujian Konfigurasi A (Baseline NLP Lokal)\n" + "-"*50)
    
    for i, data in enumerate(dataset):
        query = data["query"]
        category = data.get("category", "unknown")
        
        # Pengukuran latency 3 kali untuk stabilitas
        latencies = []
        for _ in range(3):
            start_time = time.time()
            try:
                # 1. Ekstrak intent & entitas
                ml_res = chatbot.engine.process_message(query)
                intent = ml_res.get("intent", "unknown")
                entities = ml_res.get("entities", {})
                confidence = ml_res.get("confidence", 0.0)
                
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
                
                # 2. Susun jawaban rule-based murni
                if intent in ["ask_recommendation", "ask_category", "ask_hidden_gems", "ask_unrelated"]:
                    reply_a = "Maaf, saya tidak mengerti maksud Anda."
                else:
                    raw_reply = build_response(intent, entities, query)
                    if isinstance(raw_reply, dict) and "reply" in raw_reply:
                        reply_a = raw_reply["reply"]
                    else:
                        reply_a = str(raw_reply)
                        
            except Exception as e:
                intent = "ERROR"
                entities = {}
                confidence = 0.0
                reply_a = f"ERROR: {str(e)}"
                
            latencies.append(time.time() - start_time)
            
        avg_latency = sum(latencies) / len(latencies)
        total_latency += avg_latency
        
        record = {
            "id": i + 1,
            "category": category,
            "query": query,
            "predicted_intent": intent,
            "predicted_intent_confidence": confidence,
            "entities": entities,
            "answer_A": reply_a,
            "latency_sec": round(avg_latency, 4)
        }
        
        results.append(record)
        print(f"[{i+1}/15] {query} -> {intent} | Latency: {avg_latency:.4f}s")
        
    # Simpan hasil mentah
    output_path = "tests/config_a_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"\nPengujian selesai. Hasil mentah disimpan di {output_path}")
    print(f"Rata-rata latency keseluruhan: {total_latency/len(dataset):.4f} detik")
    
    # Cetak tabel Markdown
    print("\n### Ringkasan Hasil Konfigurasi A (Baseline NLP Lokal)\n")
    print("| No | Kategori | Query | Intent | Entitas | Jawaban A | Latency (s) |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        # Bersihkan newline agar tabel markdown tidak rusak
        clean_ans = r['answer_A'].replace('\n', ' ').replace('\r', '')
        # Format entitas menjadi string rapi
        ent_str = ", ".join([f"{k}: {v}" for k, v in r['entities'].items()]) if r['entities'] else "-"
        print(f"| {r['id']} | {r['category']} | {r['query']} | {r['predicted_intent']} | {ent_str} | {clean_ans} | {r['latency_sec']} |")

if __name__ == "__main__":
    main()
