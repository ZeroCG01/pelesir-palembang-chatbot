import json
with open('tests/config_b_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
interventions = [d for d in data if d['intervensi']]
print(f"Total Interventions: {len(interventions)}")
print(f"Total PASS: {len(data) - len(interventions)}")
print(f"Average LLM Latency: {sum(d['latency_llm_detik'] for d in data)/len(data):.2f}s")
print("Failures (Interventions):")
for d in interventions:
    print(f"- Q{d['no']}: {d['jawaban_akhir'][:50]}...")
