import numpy as np
from scipy.optimize import linear_sum_assignment
from src.matching.bipartite_matcher import build_augmented_cost, match_protected_templates, match_protected_greedy

# Construct a minimal 2x2 counterexample where Greedy makes a suboptimal choice
# Suppose we have 2 query descriptors and 2 enrolled descriptors.
# Let tau = 0.5. Unmatched penalty = tau / 2 = 0.25.
# Cost of matching real pair (i, j) with distance d < tau is d.
# Distance matrix:
#        c0     c1
# r0   [0.20, 0.22]
# r1   [0.21, 0.45]
#
# Greedy 1-to-1 order:
# Smallest distance is (r0, c0) = 0.20.
# Greedy matches (r0, c0).
# Next, r0 and c0 are consumed.
# For r1, only c1 remains with d = 0.45 < tau (0.50).
# So Greedy matches (r1, c1) with d = 0.45.
# Total Greedy real distance cost = 0.20 + 0.45 = 0.65.
# Total savings = (0.50 - 0.20) + (0.50 - 0.45) = 0.30 + 0.05 = 0.35.
#
# But Hungarian:
# Can choose (r0, c1) = 0.22 and (r1, c0) = 0.21!
# Total Hungarian real distance cost = 0.22 + 0.21 = 0.43!
# Total savings = (0.50 - 0.22) + (0.50 - 0.21) = 0.28 + 0.29 = 0.57!
# Hungarian achieves a lower total augmented cost (higher savings)!

dist_matrix = np.array([
    [0.20, 0.22],
    [0.21, 0.45]
], dtype=np.float64)
tau = 0.50

augmented_cost = build_augmented_cost(dist_matrix, tau)
r_ind, c_ind = linear_sum_assignment(augmented_cost)

hungarian_pairs = [(r, c) for r, c in zip(r_ind, c_ind) if r < 2 and c < 2 and dist_matrix[r, c] < tau]
print("Hungarian matched pairs:", hungarian_pairs)
hungarian_cost = sum(dist_matrix[r, c] for r, c in hungarian_pairs)
print("Hungarian real distance sum:", hungarian_cost)

# Greedy 1-to-1:
valid_pairs = np.argwhere(dist_matrix < tau)
pair_dists = dist_matrix[valid_pairs[:, 0], valid_pairs[:, 1]]
sort_order = np.argsort(pair_dists, kind="stable")
matched_q, matched_e = set(), set()
greedy_pairs = []
for idx in sort_order:
    r, c = int(valid_pairs[idx, 0]), int(valid_pairs[idx, 1])
    if r not in matched_q and c not in matched_e:
        matched_q.add(r)
        matched_e.add(c)
        greedy_pairs.append((r, c))

print("Greedy matched pairs:", greedy_pairs)
greedy_cost = sum(dist_matrix[r, c] for r, c in greedy_pairs)
print("Greedy real distance sum:", greedy_cost)
print(f"Suboptimality gap: Hungarian cost {hungarian_cost:.2f} < Greedy cost {greedy_cost:.2f}")
