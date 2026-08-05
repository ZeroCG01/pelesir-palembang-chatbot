import subprocess
import json
import os

COMMIT = "6463b59"
FILES_TO_RESTORE = {
    "ml/data/processed/test_ner_v2.json": "ml/data/processed/test_ner_legacy_583.json",
    "ml/data/processed/train_ner_v2.json": "ml/data/processed/train_ner_legacy.json",
    "ml/data/processed/val_ner_v2.json": "ml/data/processed/val_ner_legacy.json",
}

print(f"Restoring legacy NER dataset files from Git commit {COMMIT}...")

for git_path, local_target in FILES_TO_RESTORE.items():
    try:
        content = subprocess.check_output(["git", "show", f"{COMMIT}:{git_path}"], text=True)
        data = json.loads(content)
        
        os.makedirs(os.path.dirname(local_target), exist_ok=True)
        with open(local_target, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"  ✅ Saved: {local_target} ({len(data)} sentences)")
    except Exception as e:
        print(f"  ❌ Error restoring {git_path}: {e}")

print("\nRestoration complete!")
