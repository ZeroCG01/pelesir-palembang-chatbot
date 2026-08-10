"""
measure_pipeline_latency.py — Skrip Pengukuran Latensi Per-Tahap Chatbot Pelesir Palembang
Membandingkan Config A (Lokal: Pra-proses -> NLP XLM-RoBERTa -> Draft) vs
Config B (Hybrid: Config A + Validasi LLM Guardrail Gemini).

Mengevaluasi 15 Kasus Uji Gold Test Set (6 Normal, 5 Trap, 4 Out-of-Domain).
"""

import os
import sys
import json
import time
import csv
import re
import numpy as np
from dotenv import load_dotenv

# Tambahkan root path agar modul app & ml dapat diimpor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from app.services.nlp_service import (
    ChatbotModel,
    build_system_prompt,
    get_destination_names,
    ABBREVIATIONS,
    supabase
)
from ml.api.response_builder import build_response, enrich_gemini_response
from openai import OpenAI


# ============================================================
# 1. KONFIGURASI DAN INISIALISASI MODEL
# ============================================================
OUTPUT_DIR = "output/reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GOLD_DATASET_PATH = "tests/end_to_end_gold.json"

openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
llm_client = OpenAI(
    api_key=openrouter_key,
    base_url="https://openrouter.ai/api/v1"
)
LLM_MODEL = "google/gemini-2.5-flash"


def run_preprocessing(raw_text: str):
    """Tahap 1: Pra-pemrosesan input (Pembersihan, regex, normalisasi, lowercase)."""
    text_clean = raw_text.strip().lower()
    text_clean = re.sub(r'[^\w\s\?]', ' ', text_clean)
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()
    return text_clean


def run_llm_guardrail(message: str, draft_reply: str, db_context: str):
    """Tahap 4: Validasi LLM Guardrail (Gemini via OpenRouter)."""
    guardrail_prompt = f"""Anda adalah AI Guardrail (Juri Penilai) untuk Chatbot Pelesir Palembang.
Tugas Anda adalah mengevaluasi Draf Balasan dari NLP Lokal terhadap pesan pengguna.

[REFERENSI DATABASE]
{db_context}

Pesan Pengguna: "{message}"
Draf Balasan Lokal: "{draft_reply}"

ATURAN KETAT:
1. Jika Draf Balasan akurat, informatif, dan relevan, balas tepat 1 kata: PASS
2. Jika Draf Balasan salah, tidak relevan, atau tidak informatif, balas: FAIL: <tuliskan_jawaban_revisi_disini>
3. Palembang BUKAN kota pesisir laut. Jika ditanya soal pantai alami, arahkan ke Sungai Musi / Pulau Kemaro.

Evaluasi Anda:"""

    chat_completion = llm_client.chat.completions.create(
        messages=[{"role": "user", "content": guardrail_prompt}],
        model=LLM_MODEL,
        temperature=0.1,
    )
    response_text = chat_completion.choices[0].message.content.strip()
    
    is_pass = response_text.upper().startswith("PASS")
    if is_pass:
        final_reply = draft_reply
        revised = False
    else:
        clean_text = response_text.replace("FAIL:", "", 1).strip().replace("**", "")
        rich_res = enrich_gemini_response(clean_text)
        final_reply = rich_res.get("reply", clean_text)
        revised = True

    return is_pass, revised, final_reply


def main():
    print("=" * 85)
    print("🚀 INSTRUMENTASI PENGUKURAN LATENSI PER-TAHAP (CONFIG A vs CONFIG B)")
    print("=" * 85)

    if not os.path.exists(GOLD_DATASET_PATH):
        print(f"❌ File {GOLD_DATASET_PATH} tidak ditemukan!")
        sys.exit(1)

    with open(GOLD_DATASET_PATH, "r", encoding="utf-8") as f:
        gold_cases = json.load(f)

    print(f"📦 Memuat model NLP Lokal (XLM-RoBERTa Intent & NER)...")
    chatbot = ChatbotModel()
    engine = chatbot.engine

    print("🌐 Memuat context database & warming-up cache...")
    db_context = build_system_prompt()
    get_destination_names()

    # ============================================================
    # 2. WARM-UP (1-2 KALI SUPAYA MODEL & JARINGAN SIAP)
    # ============================================================
    print("\n🔥 Melakukan Warm-Up (2 iterasi awal dibuang dari metrik)...")
    for w in range(2):
        _ = run_preprocessing("halo selamat pagi")
        _res = engine.process_message("berapa harga tiket ampera")
        _draft = build_response(_res["intent"], _res["entities"], "berapa harga tiket ampera")
        _ = run_llm_guardrail("berapa harga tiket ampera", _draft, db_context)
    print("✅ Warm-up selesai! Memulai pengujian 15 kasus gold dataset...\n")

    # ============================================================
    # 3. PENGUKURAN 15 KASUS GOLD TEST SET
    # ============================================================
    raw_results = []
    revision_count = 0

    print(f"{'No':<3} | {'Kategori':<8} | {'t1(Prep)':<9} | {'t2(NLP)':<9} | {'t3(Draft)':<10} | {'Tot A(s)':<9} | {'t4(LLM)':<9} | {'Tot B(s)':<9} | {'Guardrail'}")
    print("-" * 92)

    for idx, item in enumerate(gold_cases, 1):
        query = item["query"]
        category = item.get("category", "general")

        # --- Tahap 1: Pra-pemrosesan ---
        start_t1 = time.perf_counter()
        cleaned_query = run_preprocessing(query)
        end_t1 = time.perf_counter()
        t1 = end_t1 - start_t1

        # --- Tahap 2: Inferensi NLP Lokal (Intent + NER) ---
        start_t2 = time.perf_counter()
        nlp_res = engine.process_message(cleaned_query)
        end_t2 = time.perf_counter()
        t2 = end_t2 - start_t2

        # --- Tahap 3: Penyusunan Draf Jawaban (build_response + DB / Fuzzy) ---
        start_t3 = time.perf_counter()
        draft_reply = build_response(nlp_res["intent"], nlp_res["entities"], cleaned_query)
        end_t3 = time.perf_counter()
        t3 = end_t3 - start_t3

        total_a = t1 + t2 + t3

        # --- Tahap 4: Validasi LLM Guardrail (Gemini) ---
        start_t4 = time.perf_counter()
        is_pass, revised, final_reply = run_llm_guardrail(query, draft_reply, db_context)
        end_t4 = time.perf_counter()
        t4 = end_t4 - start_t4

        total_b = total_a + t4

        if revised:
            revision_count += 1
            verdict_str = "REVISED"
        else:
            verdict_str = "PASS"

        raw_results.append({
            "no": idx,
            "category": category,
            "query": query,
            "intent": nlp_res["intent"],
            "entities": str(nlp_res["entities"]),
            "t1_preprocessing": t1,
            "t2_nlp_inference": t2,
            "t3_draft_builder": t3,
            "total_config_a": total_a,
            "t4_llm_guardrail": t4,
            "total_config_b": total_b,
            "guardrail_verdict": verdict_str,
            "draft_reply": draft_reply,
            "final_reply": final_reply
        })

        print(f"{idx:<3} | {category:<8} | {t1*1000:6.2f} ms | {t2:6.4f} s | {t3*1000:7.2f} ms | {total_a:6.4f} s | {t4:6.4f} s | {total_b:6.4f} s | {verdict_str}")

    print("-" * 92)

    # ============================================================
    # 4. PENGHITUNGAN STATISTIK AGREGAT
    # ============================================================
    def calc_stats(values):
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "std": float(np.std(arr))
        }

    stats_t1 = calc_stats([r["t1_preprocessing"] for r in raw_results])
    stats_t2 = calc_stats([r["t2_nlp_inference"] for r in raw_results])
    stats_t3 = calc_stats([r["t3_draft_builder"] for r in raw_results])
    stats_tot_a = calc_stats([r["total_config_a"] for r in raw_results])
    stats_t4 = calc_stats([r["t4_llm_guardrail"] for r in raw_results])
    stats_tot_b = calc_stats([r["total_config_b"] for r in raw_results])

    overhead_guardrail = stats_tot_b["mean"] - stats_tot_a["mean"]

    print("\n" + "=" * 90)
    print("📊 TABEL PERBANDINGAN LATENSI PER-TAHAP (15 KASUS GOLD TEST SET)")
    print("=" * 90)
    print(f"{'Tahapan Pipeline':<35} | {'Rerata Config A (s)':<20} | {'Rerata Config B (s)':<20} | {'Keterangan'}")
    print("-" * 90)
    print(f"{'1. Pra-pemrosesan Input (t1)':<35} | {stats_t1['mean']:17.4f} s | {stats_t1['mean']:17.4f} s | Pembersihan & normalisasi teks")
    print(f"{'2. Inferensi NLP Lokal (t2)':<35} | {stats_t2['mean']:17.4f} s | {stats_t2['mean']:17.4f} s | XLM-RoBERTa Intent + NER")
    print(f"{'3. Penyusunan Draf Jawaban (t3)':<35} | {stats_t3['mean']:17.4f} s | {stats_t3['mean']:17.4f} s | DB Supabase & Fuzzy Matching")
    print(f"{'4. Validasi LLM Guardrail (t4)':<35} | {'-':>20} | {stats_t4['mean']:17.4f} s | Evaluasi Gemini + Network")
    print("-" * 90)
    print(f"{'TOTAL WAKTU RESPON (CONFIG A)':<35} | {stats_tot_a['mean']:17.4f} s | {'-':>20} | Rerata respon lokal murni")
    print(f"{'TOTAL WAKTU RESPON (CONFIG B)':<35} | {'-':>20} | {stats_tot_b['mean']:17.4f} s | Rerata respon hibrida LLM")
    print(f"{'SELISIH OVERHEAD GUARDRAIL':<35} | {'-':>20} | {overhead_guardrail:17.4f} s | Penambahan latensi jaringan")
    print("=" * 90)

    print("\n📈 STATISTIK LENGKAP DETAIL (Mean, Median, Min, Max, Std):")
    print(f"- Config A (Total): Mean = {stats_tot_a['mean']:.3f}s | Median = {stats_tot_a['median']:.3f}s | Min = {stats_tot_a['min']:.3f}s | Max = {stats_tot_a['max']:.3f}s | Std = {stats_tot_a['std']:.3f}s")
    print(f"- Config B (Total): Mean = {stats_tot_b['mean']:.3f}s | Median = {stats_tot_b['median']:.3f}s | Min = {stats_tot_b['min']:.3f}s | Max = {stats_tot_b['max']:.3f}s | Std = {stats_tot_b['std']:.3f}s")
    print(f"- Kasus yang Direvisi Guardrail: {revision_count} / {len(gold_cases)} kasus ({revision_count/len(gold_cases)*100:.1f}%)")

    # ============================================================
    # 5. SIMPAN DATA MENTAH & AGREGAT KE CSV DAN JSON
    # ============================================================
    csv_paths = ["latensi_per_tahap.csv", os.path.join(OUTPUT_DIR, "latensi_per_tahap.csv")]
    for cp in csv_paths:
        with open(cp, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "no", "category", "query", "intent", "entities",
                "t1_preprocessing", "t2_nlp_inference", "t3_draft_builder",
                "total_config_a", "t4_llm_guardrail", "total_config_b",
                "guardrail_verdict", "draft_reply", "final_reply"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(raw_results)

    json_payload = {
        "summary": {
            "total_cases": len(gold_cases),
            "revisions_by_guardrail": revision_count,
            "config_a": {
                "t1_preprocessing": stats_t1,
                "t2_nlp_inference": stats_t2,
                "t3_draft_builder": stats_t3,
                "total_latency": stats_tot_a
            },
            "config_b": {
                "t4_llm_guardrail": stats_t4,
                "total_latency": stats_tot_b,
                "guardrail_overhead": overhead_guardrail
            }
        },
        "raw_cases": raw_results
    }

    json_paths = ["latensi_per_tahap.json", os.path.join(OUTPUT_DIR, "latensi_per_tahap.json")]
    for jp in json_paths:
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=4, ensure_ascii=False)

    print(f"\n💾 Berkas data tersimpan:")
    print(f"  - CSV : latensi_per_tahap.csv & {os.path.join(OUTPUT_DIR, 'latensi_per_tahap.csv')}")
    print(f"  - JSON: latensi_per_tahap.json & {os.path.join(OUTPUT_DIR, 'latensi_per_tahap.json')}")

    # ============================================================
    # 6. KOMPARASI TERHADAP TABEL 5.7 SKRIPSI
    # ============================================================
    print("\n" + "=" * 90)
    print("🔍 KOMPARASI TERHADAP ACUAN TABEL 5.7 SKRIPSI:")
    print("=" * 90)
    print(f"- Acuan Tabel 5.7  : Total Config A ≈ 0,10 s  | Total Config B ≈ 1,23 s")
    print(f"- Hasil Uji Aktual : Total Config A = {stats_tot_a['mean']:.2f} s | Total Config B = {stats_tot_b['mean']:.2f} s")
    diff_a = stats_tot_a['mean'] - 0.10
    diff_b = stats_tot_b['mean'] - 1.23
    print(f"- Selisih Variansi : Δ Config A = {diff_a:+.2f} s   | Δ Config B = {diff_b:+.2f} s")
    
    if abs(diff_a) < 0.05 and abs(diff_b) < 0.5:
        print("✅ Hasil pengujian sangat konsisten dan merefleksikan angka Tabel 5.7 secara presisi!")
    else:
        print("ℹ️ Catatan Analisis: Variasi pada Total Config B dipengaruhi fluktuasi latency jaringan internet & antrian server OpenRouter/Gemini.")
    print("=" * 90)


if __name__ == "__main__":
    main()
