import subprocess
import os

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if res.stdout:
        print("  [OUT]", res.stdout.strip())
    if res.stderr:
        print("  [ERR]", res.stderr.strip())
    return res.returncode

print("==================================================================")
print(" LANGKAH 1: BACKUP COMMIT & PUSH KE GITHUB (BEFORE TESTS CLEANUP)")
print("==================================================================")

run_cmd("git add .")
run_cmd('git commit -m "backup: save all pre-cleanup tests scripts and experimental results"')
run_cmd("git push github main")


print("\n==================================================================")
print(" LANGKAH 2: HAPUS FILE EKSPERIMEN LAMA DI FOLDER tests/")
print("==================================================================")

files_to_remove_tests = [
    "config_a_results.json",
    "config_b_results.json",
    "config_b_results_OLD_1785343556.json",
    "end_to_end_gold.json",
    "end_to_end_output.txt",
    "evaluate_nlp_roc.py",
    "evaluate_ood.py",
    "fuzzy_threshold_eval.py",
    "run_config_a.py",
    "run_config_b.py",
    "simulasi_fuzzy_threshold.py",
    "test_end_to_end_gold.py",
    "test_zero_shot.py",
    "zero_shot_results.md"
]

for f in files_to_remove_tests:
    p = os.path.join("tests", f)
    if os.path.exists(p):
        os.remove(p)
        print(f"  ✅ Deleted: {p}")


print("\n==================================================================")
print(" LANGKAH 3: COMMIT & PUSH HASIL BERSIH KE GITHUB (AFTER TESTS CLEANUP)")
print("==================================================================")

run_cmd("git add .")
run_cmd('git commit -m "refactor: clean up obsolete test benchmarks in tests/ (keep active skripsi evaluation scripts)"')
run_cmd("git push github main")

print("\nPembersihan folder tests/ dan push ke GitHub selesai 100%!")
