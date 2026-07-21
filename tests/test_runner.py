"""
Chatbot End-to-End Test Runner
==============================
Script otomatis untuk menguji chatbot TanyaKito secara end-to-end.
Mengirim skenario percakapan ke API dan memverifikasi respons.

Cara pakai:
  python tests/test_runner.py                          # Test ke Hugging Face (production)
  python tests/test_runner.py --url http://localhost:8000  # Test ke server lokal
  python tests/test_runner.py --category sidang_scenario   # Test kategori tertentu saja
  python tests/test_runner.py --id TC-200                  # Test satu test case saja
"""

import json
import sys
import os
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime

# ========== KONFIGURASI ==========
DEFAULT_API_URL = "https://zerocg-pelesir-palembang-chatbot.hf.space"
TEST_SUITE_PATH = Path(__file__).parent / "test_suite.json"
DELAY_BETWEEN_TURNS = 15.0   # detik antar turn (Maksimal 4 request/menit, sangat aman dari limit 15/menit Gemini Free)
DELAY_BETWEEN_TESTS = 20.0   # detik antar test case (Memberi napas panjang pada server Gemini)


# ========== WARNA TERMINAL ==========
class Color:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


def send_chat(api_url: str, message: str, history: list) -> str:
    """Kirim pesan ke API chatbot dan kembalikan balasannya. Retry jika 500/429."""
    MAX_RETRIES = 2
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(
                f"{api_url}/api/chat",
                json={"message": message, "history": history},
                headers={"Content-Type": "application/json"},
                timeout=120  # Timeout lebih lama karena server retry Gemini internal
            )
            # Retry jika 500 atau 429 (rate limit di server)
            if response.status_code in [429, 500] and attempt < MAX_RETRIES:
                wait = 20 * (attempt + 1)
                print(f"  {Color.YELLOW}⏳ Server error {response.status_code}, retry {attempt+1}/{MAX_RETRIES} setelah {wait}s...{Color.END}")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json().get("reply", "")
        except requests.exceptions.Timeout:
            return "[ERROR: TIMEOUT - Server tidak merespons dalam 120 detik]"
        except requests.exceptions.ConnectionError:
            return "[ERROR: CONNECTION - Tidak bisa terhubung ke server]"
        except Exception as e:
            if attempt < MAX_RETRIES and ("500" in str(e) or "429" in str(e)):
                wait = 20 * (attempt + 1)
                print(f"  {Color.YELLOW}⏳ Error: {str(e)[:60]}, retry {attempt+1}/{MAX_RETRIES} setelah {wait}s...{Color.END}")
                time.sleep(wait)
                continue
            return f"[ERROR: {str(e)}]"


def check_response(reply: str, must_contain: list, must_not: list) -> tuple:
    """
    Periksa apakah respons memenuhi kriteria.
    Returns: (passed: bool, failures: list[str])
    """
    failures = []
    reply_lower = reply.lower()

    for keyword in must_contain:
        if keyword.lower() not in reply_lower:
            failures.append(f"  ❌ HARUS mengandung '{keyword}' tapi TIDAK ditemukan")

    for keyword in must_not:
        if keyword.lower() in reply_lower:
            failures.append(f"  ❌ TIDAK BOLEH mengandung '{keyword}' tapi DITEMUKAN")

    return (len(failures) == 0, failures)


def run_test_case(api_url: str, test_case: dict) -> dict:
    """
    Jalankan satu test case (bisa multi-turn).
    Returns: dict dengan hasil per turn dan status keseluruhan.
    """
    tc_id = test_case["id"]
    description = test_case["description"]
    turns = test_case["turns"]
    
    history = []  # Simulasi riwayat percakapan
    turn_results = []
    all_passed = True

    for i, turn in enumerate(turns):
        user_input = turn["input"]
        must_contain = turn.get("response_must_contain", [])
        must_not = turn.get("response_must_not", [])

        # Kirim ke API
        reply = send_chat(api_url, user_input, history)

        # Cek error koneksi
        if reply.startswith("[ERROR"):
            turn_results.append({
                "turn": i + 1,
                "input": user_input,
                "reply": reply,
                "passed": False,
                "failures": [f"  ❌ {reply}"]
            })
            all_passed = False
            break

        # Validasi respons
        passed, failures = check_response(reply, must_contain, must_not)
        if not passed:
            all_passed = False

        turn_results.append({
            "turn": i + 1,
            "input": user_input,
            "reply": reply[:200] + "..." if len(reply) > 200 else reply,
            "passed": passed,
            "failures": failures
        })

        # Update history untuk turn berikutnya
        history.append({"role": "user", "content": user_input})
        history.append({"role": "model", "content": reply})

        if i < len(turns) - 1:
            time.sleep(DELAY_BETWEEN_TURNS)

    return {
        "id": tc_id,
        "description": description,
        "category": test_case.get("category", "unknown"),
        "passed": all_passed,
        "turn_results": turn_results
    }


def print_result(result: dict):
    """Cetak hasil satu test case ke terminal."""
    status = f"{Color.GREEN}✅ PASS{Color.END}" if result["passed"] else f"{Color.RED}❌ FAIL{Color.END}"
    print(f"\n{Color.BOLD}[{result['id']}]{Color.END} {status} - {result['description']}")

    for tr in result["turn_results"]:
        turn_status = "✅" if tr["passed"] else "❌"
        print(f"  {Color.DIM}Turn {tr['turn']}:{Color.END} \"{tr['input']}\"")
        print(f"  {Color.DIM}Reply:{Color.END} {tr['reply'][:150]}{'...' if len(tr['reply']) > 150 else ''}")
        
        if not tr["passed"]:
            for f in tr["failures"]:
                print(f"  {Color.RED}{f}{Color.END}")


def generate_report(results: list, api_url: str) -> str:
    """Generate laporan ringkasan dalam format teks."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    # Hitung per kategori
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0, "failed": 0}
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["passed"] += 1
        else:
            categories[cat]["failed"] += 1

    report = []
    report.append("=" * 60)
    report.append(f"  CHATBOT TEST REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"  Target: {api_url}")
    report.append("=" * 60)
    report.append(f"\n  Total: {total}  |  Passed: {passed}  |  Failed: {failed}  |  Rate: {passed/total*100:.1f}%\n")

    report.append("  Per Kategori:")
    report.append("  " + "-" * 56)
    report.append(f"  {'Kategori':<30} {'Total':>5} {'Pass':>5} {'Fail':>5} {'Rate':>7}")
    report.append("  " + "-" * 56)
    for cat, stats in sorted(categories.items()):
        rate = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
        marker = "✅" if stats["failed"] == 0 else "❌"
        report.append(f"  {marker} {cat:<28} {stats['total']:>5} {stats['passed']:>5} {stats['failed']:>5} {rate:>6.1f}%")
    report.append("  " + "-" * 56)

    if failed > 0:
        report.append("\n  Test Case yang GAGAL:")
        for r in results:
            if not r["passed"]:
                report.append(f"  ❌ [{r['id']}] {r['description']}")
                for tr in r["turn_results"]:
                    if not tr["passed"]:
                        report.append(f"     Turn {tr['turn']}: \"{tr['input']}\"")
                        for f in tr["failures"]:
                            report.append(f"     {f}")

    report.append("\n" + "=" * 60)
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Chatbot End-to-End Test Runner")
    parser.add_argument("--url", default=DEFAULT_API_URL, help="URL API chatbot")
    parser.add_argument("--category", default=None, help="Filter berdasarkan kategori (misal: sidang_scenario)")
    parser.add_argument("--id", default=None, help="Jalankan satu test case saja berdasarkan ID (misal: TC-200)")
    parser.add_argument("--save", default=None, help="Simpan laporan ke file (misal: report.txt)")
    args = parser.parse_args()

    # Load test suite
    with open(TEST_SUITE_PATH, "r", encoding="utf-8") as f:
        test_suite = json.load(f)

    # Filter jika diperlukan
    if args.id:
        test_suite = [tc for tc in test_suite if tc["id"] == args.id]
        if not test_suite:
            print(f"{Color.RED}Test case '{args.id}' tidak ditemukan!{Color.END}")
            sys.exit(1)
    elif args.category:
        test_suite = [tc for tc in test_suite if tc["category"] == args.category]
        if not test_suite:
            print(f"{Color.RED}Kategori '{args.category}' tidak ditemukan!{Color.END}")
            sys.exit(1)

    print(f"\n{Color.BOLD}{Color.CYAN}🤖 Chatbot End-to-End Test Runner{Color.END}")
    print(f"{Color.DIM}Target: {args.url}{Color.END}")
    print(f"{Color.DIM}Test Cases: {len(test_suite)}{Color.END}")
    print(f"{Color.DIM}{'=' * 50}{Color.END}")

    # Cek koneksi dulu
    print(f"\n{Color.YELLOW}🔌 Mengecek koneksi ke server...{Color.END}")
    try:
        r = requests.get(args.url, timeout=15)
        print(f"{Color.GREEN}✅ Server aktif!{Color.END}")
    except Exception:
        print(f"{Color.RED}❌ Server tidak bisa dihubungi di {args.url}{Color.END}")
        print(f"{Color.YELLOW}💡 Pastikan server sudah berjalan, atau coba: --url http://localhost:8000{Color.END}")
        sys.exit(1)

    # Jalankan semua test
    results = []
    for i, tc in enumerate(test_suite):
        print(f"\n{Color.DIM}--- Running {i+1}/{len(test_suite)}: [{tc['id']}] ---{Color.END}")
        result = run_test_case(args.url, tc)
        results.append(result)
        print_result(result)
        
        if i < len(test_suite) - 1:
            time.sleep(DELAY_BETWEEN_TESTS)

    # Cetak laporan
    report = generate_report(results, args.url)
    print(f"\n{report}")

    # Simpan laporan jika diminta
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n{Color.GREEN}📄 Laporan disimpan ke: {args.save}{Color.END}")

    # Exit code berdasarkan hasil
    failed_count = sum(1 for r in results if not r["passed"])
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
