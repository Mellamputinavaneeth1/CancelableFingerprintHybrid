import json

with open("experiments/results/unlinkability/unlinkability_report.json", "r", encoding="utf-8") as f:
    unl = json.load(f)

for k, v in unl.items():
    if isinstance(v, (dict, list)):
        print(f"[{k}]: {json.dumps(v, indent=2)}")
    else:
        print(f"{k}: {v}")
