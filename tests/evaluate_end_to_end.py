import os
import sys
import json
import time
from dotenv import load_dotenv

# Ensure parent dir is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from app.services.nlp_service import ChatbotModel, build_system_prompt
from ml.api.response_builder import build_response
import google.generativeai as genai

def evaluate_with_llm_judge(query, response, context):
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""Anda adalah Penilai Evaluasi Independen (LLM-as-a-Judge) untuk eksperimen NLP Chatbot Pariwisata.

Pertanyaan Pengguna: "{query}"
Jawaban Sistem: "{response}"

Konteks Aturan & Database (Untuk Referensi Anda):
{context[:1500]}...

Tugas Anda:
1. Nilai CORRECTNESS: Berikan angka 1 jika jawaban BENAR (menjawab sesuai data, atau menolak menjawab hal di luar konteks pariwisata dengan sopan). Berikan angka 0 jika salah kaprah, gagal memberikan rekomendasi yang masuk akal, atau gagal merespons sesuai intent asli.
2. Nilai HALLUCINATION: Berikan angka 1 jika sistem MENGARANG informasi palsu/destinasi palsu/harga palsu. Berikan angka 0 jika sistem AMAN (tidak berhalusinasi, termasuk jika ia menolak menjawab karena data tidak ada).

PENTING: Jawab HANYA dengan format JSON persis seperti ini:
{{"correctness": 1, "hallucination": 0}}
"""
    try:
        chat = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        return json.loads(chat.text.strip())
    except Exception as e:
        print(f"Error evaluating: {e}")
        return {"correctness": 0, "hallucination": 0}

def main():
    print("Loading Dataset...")
    with open("tests/end_to_end_gold.json", "r") as f:
        dataset = json.load(f)
        
    print("Initializing Models...")
    chatbot = ChatbotModel()
    db_context = build_system_prompt()
    
    results = []
    os.environ["LLM_PROVIDER"] = "gemini" # PAKSA GEMINI UNTUK GUARDRAIL

    
    config_a_stats = {"latency": [], "correctness": 0, "hallucination": 0}
    config_b_stats = {"latency": [], "correctness": 0, "hallucination": 0, "guardrail_interventions": 0}
    
    for i, data in enumerate(dataset):
        query = data["query"]
        category = data["category"]
        print(f"\n[{i+1}/{len(dataset)}] Evaluating: {query}")
        
        # --- CONFIG A: NLP ONLY (No Guardrail) ---
        start_a = time.time()
        # Manual bypass simulating old architecture
        ml_res = chatbot.engine.process_message(query)
        intent = ml_res["intent"]
        entities = ml_res["entities"]
        
        # Catch generative intents manually like the old code did (which failed without LLM)
        if intent in ["ask_recommendation", "ask_category", "ask_hidden_gems", "ask_unrelated"]:
            reply_a_text = "Maaf, saya tidak mengerti maksud Anda." # Typical fallback without generative AI
        else:
            reply_a = build_response(intent, entities, query)
            if isinstance(reply_a, dict) and "reply" in reply_a:
                reply_a_text = reply_a["reply"]
            else:
                reply_a_text = str(reply_a)
                
        latency_a = time.time() - start_a
        config_a_stats["latency"].append(latency_a)
        
        # --- CONFIG B: FULL PIPELINE (NLP + Guardrail) ---
        start_b = time.time()
        reply_b = chatbot.generate_reply(query, [])
        if isinstance(reply_b, dict):
            reply_b_text = reply_b.get("reply", str(reply_b))
            source_b = reply_b.get("source", "lokal")
        else:
            reply_b_text = str(reply_b)
            source_b = "lokal"
            
        latency_b = time.time() - start_b
        config_b_stats["latency"].append(latency_b)
        
        if source_b != "lokal" and source_b != "lokal_bypass":
            config_b_stats["guardrail_interventions"] += 1
            print("  >> Guardrail INTERVENED!")
            
        # --- JUDGE EVALUATION ---
        judge_a = evaluate_with_llm_judge(query, reply_a_text, db_context)
        judge_b = evaluate_with_llm_judge(query, reply_b_text, db_context)
        
        config_a_stats["correctness"] += judge_a.get("correctness", 0)
        config_a_stats["hallucination"] += judge_a.get("hallucination", 0)
        
        config_b_stats["correctness"] += judge_b.get("correctness", 0)
        config_b_stats["hallucination"] += judge_b.get("hallucination", 0)
        
        results.append({
            "query": query,
            "category": category,
            "reply_a": reply_a_text,
            "reply_b": reply_b_text,
            "judge_a": judge_a,
            "judge_b": judge_b,
            "guardrail_triggered": source_b != "lokal"
        })
        
        time.sleep(1) # avoid rate limits
        
    print("\n\n==== HASIL EVALUASI E2E ====")
    n = len(dataset)
    print(f"Total Dataset: {n}")
    
    avg_lat_a = sum(config_a_stats["latency"]) / n
    avg_lat_b = sum(config_b_stats["latency"]) / n
    
    acc_a = config_a_stats["correctness"] / n * 100
    acc_b = config_b_stats["correctness"] / n * 100
    
    hal_a = config_a_stats["hallucination"] / n * 100
    hal_b = config_b_stats["hallucination"] / n * 100
    
    print("\n| Metrik | Config A (NLP Saja) | Config B (NLP + Guardrail) |")
    print("|--------|---------------------|----------------------------|")
    print(f"| Answer Correctness | {acc_a:.1f}% | {acc_b:.1f}% |")
    print(f"| Hallucination Rate | {hal_a:.1f}% | {hal_b:.1f}% |")
    print(f"| Avg Latency | {avg_lat_a:.2f}s | {avg_lat_b:.2f}s |")
    print(f"| Guardrail Interventions | 0 | {config_b_stats['guardrail_interventions']} |")
    
    with open("tests/e2e_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nHasil detail disimpan di tests/e2e_results.json")

if __name__ == "__main__":
    main()
