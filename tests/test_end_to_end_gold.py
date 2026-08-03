import asyncio
import json
import os
import sys

# Tambahkan path root ke sys.path agar import app.services berhasil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.nlp_service import ChatbotModel

def main():
    print("Memuat Dataset Gold...")
    with open("tests/end_to_end_gold.json", "r", encoding="utf-8") as f:
        gold_data = json.load(f)
        
    print("Memuat Chatbot Model (NLP + Guardrail)...")
    nlp_model = ChatbotModel()
    
    print("\nMengeksekusi Uji Coba End-to-End...\n")
    
    # Header tabel
    print("| No | Kategori | Query | Perilaku Diharapkan | Output A (NLP saja) | A Benar? | Intervensi Guardrail? | Output B (NLP+Guardrail) | B Benar? |")
    print("|---|---|---|---|---|---|---|---|---|")
    
    original_eval = nlp_model.evaluate_with_guardrail
    
    for i, d in enumerate(gold_data):
        q = d["query"]
        cat = d["category"]
        expected = d["expected_behavior"]
        
        captured_draft = ""
        
        # Override sementara untuk menangkap draf lokal (Output A)
        def mock_eval(msg, draft, hist):
            nonlocal captured_draft
            captured_draft = draft
            # Panggil fungsi aslinya untuk mendapatkan Output B
            return original_eval(msg, draft, hist)
            
        nlp_model.evaluate_with_guardrail = mock_eval
        
        # Jalankan keseluruhan pipeline
        try:
            final_resp = nlp_model.generate_reply(q, [])
            out_b = final_resp.get("reply", str(final_resp))
            source = final_resp.get("source", "")
        except Exception as e:
            out_b = f"ERROR: {e}"
            source = "error"
            
        # Kembalikan ke fungsi asli
        nlp_model.evaluate_with_guardrail = original_eval
        
        # Pembersihan teks agar rapi di tabel markdown
        out_a = captured_draft.replace("\n", " ").replace("|", "\|").strip()
        out_b = out_b.replace("\n", " ").replace("|", "\|").strip()
        
        if len(out_a) > 80: out_a = out_a[:77] + "..."
        if len(out_b) > 80: out_b = out_b[:77] + "..."
        
        intervensi = "Ya" if source != "lokal" else "Tidak"
        
        # Tampilkan kosong dulu untuk "Benar?" agar user / kita bisa review manual
        print(f"| {i+1} | {cat} | {q} | {expected} | {out_a} | ❓ | {intervensi} | {out_b} | ❓ |")

if __name__ == "__main__":
    main()
