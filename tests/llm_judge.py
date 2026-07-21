import os
import json
import time
import requests
import argparse
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY tidak ditemukan di .env")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)
# Gunakan Gemini 2.5 Flash
model = genai.GenerativeModel('gemini-2.5-flash')

parser = argparse.ArgumentParser(description="LLM-as-a-Judge Tester")
parser.add_argument("--url", default="http://localhost:8000", help="URL API Chatbot (Tanpa /api/chat)")
parser.add_argument("--file", default="dosen_killer.json", help="File JSON berisi soal uji")
args = parser.parse_args()

API_URL = f"{args.url.rstrip('/')}/api/chat"

def judge_response(question, expected_intent, bot_reply):
    """Meminta Gemini bertindak sebagai Juri untuk menilai kualitas jawaban bot"""
    prompt = f"""
    Kamu adalah Juri Evaluasi Chatbot (LLM-as-a-Judge).
    Tugasmu adalah menilai apakah jawaban chatbot sudah tepat dan relevan dengan pertanyaan pengguna.
    
    Pertanyaan Pengguna: "{question}"
    Ekspektasi Intent: {expected_intent}
    Jawaban Chatbot: "{bot_reply}"
    
    Berikan penilaian objektif berdasarkan kriteria:
    1. Apakah jawaban menjawab inti pertanyaan? (Skor tinggi jika ya)
    2. Apakah chatbot salah paham / salah konteks? (Misal ditanya rute, tapi dikasih alamat -> Skor rendah)
    3. Apakah informasi yang diberikan berguna?
    
    Keluarkan WAJIB dalam format JSON saja:
    {{
      "score": <angka_1_sampai_10>,
      "reasoning": "<alasan_singkat_mengapa_diberi_skor_tersebut>"
    }}
    """
    
    from google.api_core.exceptions import ResourceExhausted
    
    while True:
        try:
            res = model.generate_content(prompt)
            text_output = res.text.strip()
            if text_output.startswith("```json"):
                text_output = text_output.replace("```json", "", 1)
            if text_output.endswith("```"):
                text_output = text_output[:-3]
            
            return json.loads(text_output)
        except ResourceExhausted:
            print("  ⏳ [JURI] Terkena limit API. Juri sedang istirahat 35 detik...")
            time.sleep(35)
        except Exception as e:
            return {"score": 0, "reasoning": f"Gagal mengevaluasi: {e}"}

def run_evaluation():
    json_path = os.path.join(os.path.dirname(__file__), args.file)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print(f"❌ File {args.file} tidak ditemukan!")
        return

    print(f"⚖️  [LLM-AS-A-JUDGE] Memulai evaluasi semantik untuk {len(test_cases)} soal...\n")
    
    total_score = 0
    passed = 0
    failed = 0
    
    for i, tc in enumerate(test_cases):
        q = tc.get("question", "")
        expected_intent = tc.get("expected_intent", "")
        
        print(f"[{i+1}/{len(test_cases)}] 👤 User: '{q}'")
        
        # 1. Tanya ke Chatbot
        try:
            res = requests.post(API_URL, json={"message": q}, timeout=15)
            if res.status_code == 200:
                data = res.json()
                bot_reply = data.get("reply", "")
                
                # 2. Minta Juri Menilai
                evaluation = judge_response(q, expected_intent, bot_reply)
                score = evaluation.get("score", 0)
                reasoning = evaluation.get("reasoning", "")
                
                total_score += score
                
                formatted_reply = bot_reply.replace('\n', '\n               ')
                print(f"         🤖 Bot : {formatted_reply}")
                
                if score >= 7:
                    passed += 1
                    print(f"         ✅ [JURI: {score}/10] {reasoning}\n")
                else:
                    failed += 1
                    print(f"         ❌ [JURI: {score}/10] {reasoning}\n")
            else:
                print(f"         ⚠️ Error dari server chatbot: {res.status_code}\n")
        except requests.exceptions.RequestException as e:
            print(f"         ⚠️ Gagal menghubungi chatbot: {e}\n")
            
        # Jeda sebentar agar tidak spam request ke Juri
        time.sleep(2)
        
    print("==================================================")
    print("🏆 REKAP HASIL EVALUASI SEMANTIK (LLM-AS-A-JUDGE) 🏆")
    print("==================================================")
    print(f"Total Soal Diuji  : {len(test_cases)}")
    print(f"Rata-rata Skor    : {total_score / len(test_cases):.1f} / 10.0")
    print(f"Jawaban Relevan   : {passed} (Skor >= 7)")
    print(f"Jawaban Meleset   : {failed} (Skor < 7)")
    print("==================================================")

if __name__ == "__main__":
    run_evaluation()
