import os
import json
import time
import argparse
import requests
import google.generativeai as genai
from dotenv import load_dotenv

# ========== KONFIGURASI ==========
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("❌ Error: GEMINI_API_KEY tidak ditemukan di file .env")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)
# Gunakan Gemini 2.5 Flash (Kapasitas gratis lebih terjamin di akun ini)
model = genai.GenerativeModel('gemini-2.5-flash')

parser = argparse.ArgumentParser(description="Dosen Killer - Adversarial Tester")
parser.add_argument("--url", default="http://localhost:8000", help="URL API Chatbot (Tanpa /api/chat)")
parser.add_argument("--count", type=int, default=20, help="Jumlah pertanyaan yang di-generate")
args = parser.parse_args()

API_URL = f"{args.url.rstrip('/')}/api/chat"

# Daftar Intent yang didukung NLP Lokal
INTENTS = [
    "ask_destination_info",
    "ask_operating_hours",
    "ask_ticket_price",
    "ask_facilities",
    "ask_route",
    "ask_location_access",
    "ask_recommendation"
]

def generate_killer_questions(total_count=20):
    """
    Karena API Key pengguna mengalami limit 0 (habis kuota) untuk semua model free tier,
    kita akan menggunakan kumpulan soal 'Dosen Killer' yang sudah di-generate secara offline.
    """
    import os
    json_path = os.path.join(os.path.dirname(__file__), "dosen_killer.json")
    
    print(f"👨‍🏫 [DOSEN KILLER] Memuat soal ujian ekstrem dari file offline...")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            all_questions = json.load(f)
            
        # Potong sesuai jumlah yang diminta (maksimal yang tersedia)
        selected = all_questions[:total_count]
        print(f"✅ Berhasil memuat {len(selected)} soal mematikan (Bebas Limit API)!\n")
        return selected
    except FileNotFoundError:
        print("❌ File dosen_killer.json tidak ditemukan!")
        exit(1)

def test_chatbot(test_cases):
    """Menembakkan pertanyaan ke chatbot lokal dan menilai ketahanannya"""
    print(f"🚀 Memulai Ujian Sidang ke {API_URL}...\n")
    
    local_success = 0
    gemini_fallback = 0
    errors = 0
    
    report = []
    
    for i, tc in enumerate(test_cases):
        q = tc["question"]
        expected = tc["expected_intent"]
        reason = tc.get("reasoning", "")
        
        print(f"[{i+1}/{len(test_cases)}] 📝 Soal  : '{q}'")
        print(f"         🎯 Target: {expected} ({reason})")
        
        try:
            # Simulasi history ringan untuk follow-up questions
            mock_history = [
                {"role": "user", "content": "ceritain tentang Jembatan Ampera"},
                {"role": "assistant", "content": "Jembatan Ampera adalah ikon kota Palembang..."}
            ]
            
            res = requests.post(
                API_URL, 
                json={"message": q, "history": mock_history},
                timeout=15
            )
            
            if res.status_code == 200:
                data = res.json()
                source = data.get("source", "unknown")
                reply = data.get("reply", "")
                
                # Format reply agar rapi (jika ada baris baru, indentasi)
                formatted_reply = reply.replace('\n', '\n                    ')
                
                # Evaluasi: Jika source == "lokal", berarti ML/Regex kita berhasil menangkapnya
                if source == "lokal" or source == "local":
                    print(f"         ✅ [LULUS NLP LOKAL]")
                    print(f"            🤖 Bot: {formatted_reply}\n")
                    local_success += 1
                else:
                    print(f"         ⚠️  [FALLBACK GEMINI]")
                    print(f"            🤖 Bot: {formatted_reply}\n")
                    gemini_fallback += 1
                    report.append({"question": q, "reason": "Jatuh ke Gemini Fallback"})
            else:
                print(f"         ❌ [ERROR HTTP {res.status_code}]")
                errors += 1
                report.append({"question": q, "reason": f"HTTP Error {res.status_code}"})
                
        except requests.exceptions.ConnectionError:
            print(f"  ⚠️  Gagal terhubung ke {API_URL}. Pastikan server menyala.")
            exit(1)
            
        print("-" * 50)
        time.sleep(1) 
        
    print("\n" + "="*50)
    print("🏆 REKAP NILAI UJIAN SIDANG DARI DOSEN KILLER 🏆")
    print("="*50)
    print(f"Total Soal Ujian              : {len(test_cases)}")
    print(f"Berhasil Ditangani NLP Lokal  : {local_success} (Sangat Bagus!)")
    print(f"Gagal Lokal (Dilempar ke LLM) : {gemini_fallback} (Perlu perbaikan Regex/Retrain ML)")
    print(f"Error Server                  : {errors}")
    
    accuracy = (local_success / len(test_cases)) * 100
    print(f"\nSKOR KETAHANAN LOKAL: {accuracy:.1f} / 100")
    
    if accuracy >= 80:
        print("Kesan Dosen: 'Luar biasa, sistem kamu sangat tangguh menghadapi user bar-bar!'")
    elif accuracy >= 50:
        print("Kesan Dosen: 'Cukup baik, tapi masih banyak bocor ke LLM. Tingkatkan dataset ML kamu.'")
    else:
        print("Kesan Dosen: 'Sistem NLP lokal kamu masih rentan. Revisi dataset dan regex!'")

    if len(report) > 0:
        print("\n" + "-"*50)
        print("🚨 PERTANYAAN YANG GAGAL DI-HANDLE LOKAL (Jatuh ke Gemini):")
        for idx, item in enumerate(report):
            print(f" {idx+1}. {item['question']}")
        print("-"*50)

if __name__ == "__main__":
    test_cases = generate_killer_questions(args.count)
    test_chatbot(test_cases)
