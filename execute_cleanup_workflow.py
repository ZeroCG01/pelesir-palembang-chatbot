import subprocess
import shutil
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
print(" LANGKAH 1: SAVE BACKUP DENGAN COMMIT & PUSH KE GITHUB (BEFORE CLEANUP)")
print("==================================================================")

run_cmd("git add .")
run_cmd('git commit -m "backup: save all pre-cleanup state, legacy datasets, and temporary scripts"')
run_cmd("git push github main")


print("\n==================================================================")
print(" LANGKAH 2: HAPUS FILE TEMPORARY / SULIT DIKASIH PROSES PEMBERSIHAN")
print("==================================================================")

# 1. Duplicate nested directory
nested_dir = os.path.join("ml", "data", "ml")
if os.path.exists(nested_dir):
    shutil.rmtree(nested_dir)
    print(f"  ✅ Deleted duplicate nested folder: {nested_dir}")

# 2. Files in ml/data/processed/
files_to_remove_processed = [
    "intents_augmented.csv",
    "intents_augmented_v2.csv",
    "test_intents.csv",
    "test_intents_v2_dedup.csv",
    "train_intents.csv",
    "train_intents_raw_v2.csv",
    "val_intents.csv",
    "test_ner.json",
    "test_ner_v2.json",
    "test_ner_seen.json",
    "test_ner_holdout.json",
    "train_ner.json",
    "train_ner_v2.json",
    "train_ner_augmented.json",
    "train_ner_augmented_v2.json",
    "val_ner.json",
    "val_ner_v2.json"
]

for f in files_to_remove_processed:
    p = os.path.join("ml", "data", "processed", f)
    if os.path.exists(p):
        os.remove(p)
        print(f"  ✅ Deleted file: {p}")

# 3. Files in ml/data/raw/
files_to_remove_raw = [
    "intents_bilingual.csv",
    "intents_bilingual_v2.csv",
    "ner_dataset.json",
    "ner_dataset_v2.json"
]

for f in files_to_remove_raw:
    p = os.path.join("ml", "data", "raw", f)
    if os.path.exists(p):
        os.remove(p)
        print(f"  ✅ Deleted file: {p}")

# 4. Obsolete scripts in ml/data/
scripts_to_remove = [
    "audit_and_proxy_eval.py",
    "augment_data.py",
    "find_leak.py",
    "get_final_stage2_validation_raw.py",
    "get_final_verification_reports.py",
    "get_raw_reports.py",
    "get_stage2_preflight_reports.py",
    "get_user_requested_reports.py",
    "post_dedup_ner_and_intent.py",
    "restore_legacy_ner.py",
    "run_augmented_intent_dedup_eval.py",
    "run_complete_pipeline_report.py",
    "run_final_gate_checks.py",
    "run_stage1_final_check.py",
    "run_unrelated_check.py",
    "split_data.py"
]

for s in scripts_to_remove:
    p = os.path.join("ml", "data", s)
    if os.path.exists(p):
        os.remove(p)
        print(f"  ✅ Deleted script: {p}")


print("\n==================================================================")
print(" LANGKAH 3: COMMIT & PUSH HASIL BERSIH KE GITHUB (AFTER CLEANUP)")
print("==================================================================")

run_cmd("git add .")
run_cmd('git commit -m "refactor: clean up obsolete datasets and temporary scripts (Intent v2 & NER 583 legacy baseline)"')
run_cmd("git push github main")

print("\nPembersihan dan push ke GitHub selesai 100%!")
