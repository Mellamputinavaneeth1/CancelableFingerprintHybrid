import numpy as np
from src.evaluation.independent_evaluator import compute_independent_metrics, independent_confusion_matrix
from src.evaluation.metrics import compute_roc_metrics

g = np.load("experiments/results/fvc2002/DB1_A_9D_delaunay/prot_genuine_scores.npy")
i = np.load("experiments/results/fvc2002/DB1_A_9D_delaunay/prot_impostor_scores.npy")

indep = compute_independent_metrics(g, i)
primary = compute_roc_metrics(g, i, warn_on_inversion=False)

print(f"DB1_A EER:    Independent={indep['eer_pct']:.2f}%, Primary={primary['eer']*100:.2f}%")
print(f"DB1_A AUC:    Independent={indep['auc']:.4f}, Primary={primary['auc']:.4f}")
print(f"DB1_A TAR@1%: Independent={indep['tar_at_1pct_far_pct']:.2f}%, Primary={primary['tar_at_1pct_far']*100:.2f}%")

conf = independent_confusion_matrix(g, i, threshold=0.3751)
print(f"Confusion at 0.3751: TP={conf['tp']}, FN={conf['fn']}, FP={conf['fp']}, TN={conf['tn']}")
assert conf['tp'] == 1654 and conf['fn'] == 1146
assert conf['fp'] == 2050 and conf['tn'] == 2900
print("Exact confusion matches baseline report bit-for-bit!")
