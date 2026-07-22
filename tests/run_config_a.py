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
