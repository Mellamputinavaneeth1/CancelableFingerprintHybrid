import json

with open("experiments/results/manifest_index.json") as f:
    idx = json.load(f)

print(f"Total manifests: {idx['total_manifests']}")
print(f"Total trials accounted: {idx['total_trials_accounted']}")
print(f"Total genuine: {idx['total_genuine_trials']}, impostor: {idx['total_impostor_trials']}")
print(f"Confusion totals: {idx['confusion_totals']}")
print("-" * 120)
print(f"{'ID':2s} | {'Manifest Name':35s} | {'Family':12s} | {'Gen':5s} | {'Imp':5s} | {'Total':5s} | {'EER%':6s} | {'AUC':6s} | {'TP':5s} | {'FN':5s} | {'FP':5s} | {'TN':5s} | {'Threshold'}")
print("-" * 120)
for m in idx["manifests"]:
    mid = m["manifest_id"]
    name = m["manifest_name"]
    fam = m["database_family"]
    gen = m["n_genuine"]
    imp = m["n_impostor"]
    tot = m["total_trials"]
    eer = m["eer_pct"]
    auc = m["auc"]
    tp = m["tp"]
    fn = m["fn"]
    fp = m["fp"]
    tn = m["tn"]
    th = m["eval_threshold"]
    print(f"{mid:2d} | {name:35s} | {fam:12s} | {gen:5d} | {imp:5d} | {tot:5d} | {eer:6.2f} | {auc:6.4f} | {tp:5d} | {fn:5d} | {fp:5d} | {tn:5d} | {th:.4f}")
