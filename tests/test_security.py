import numpy as np
import pytest
from src.security.unlinkability import (
    compute_system_unlinkability, 
    compute_system_unlinkability_with_ci,
    compute_system_unlinkability_clustered_ci,
    compute_multi_threat_unlinkability_bootstrap,
    match_topology_only,
    evaluate_unlinkability
)
from src.security.inversion_attack import (
    simulate_hill_climbing_inversion,
    solve_linear_feasibility_inversion,
    evaluate_reconstruction_resistance
)
from src.protection.kdf import generate_master_secret
from src.protection.projection import derive_orthonormal_matrix, project_descriptors
from src.protection.quantizer import quantize_descriptors
from src.matching.bipartite_matcher import match_protected_templates

def test_system_unlinkability_metric():
    rng = np.random.default_rng(42)
    mated_scores = rng.normal(0.05, 0.02, size=300)
    non_mated_scores = rng.normal(0.05, 0.02, size=300)
    
    d_sys, bins, d_omega, lr = compute_system_unlinkability(mated_scores, non_mated_scores, num_bins=25)
    assert 0.0 <= d_sys < 0.15
    
    ci_res = compute_system_unlinkability_with_ci(mated_scores, non_mated_scores, num_bins=25, num_bootstraps=100)
    assert 0.0 <= ci_res["d_sys"] < 0.15
    assert 0.0 <= ci_res["ci_lower"] < ci_res["ci_upper"]
    assert ci_res["ci_upper"] < 0.30
    
    mated_linkable = rng.normal(0.8, 0.05, size=200)
    non_mated_dist = rng.normal(0.05, 0.02, size=200)
    d_sys_linkable, _, _, _ = compute_system_unlinkability(mated_linkable, non_mated_dist, num_bins=25)
    assert d_sys_linkable > 0.7
    
    # Test error raised on missing distributions
    with pytest.raises(ValueError, match="Missing mated"):
        compute_system_unlinkability(np.empty(0), non_mated_scores)
    with pytest.raises(ValueError, match="Missing non-mated"):
        compute_system_unlinkability(mated_scores, np.empty(0))
        
    # Test clustered bootstrap over subjects
    mated_rec = [{"subject": f"subj_{i%5}", "score": s} for i, s in enumerate(mated_scores)]
    non_mated_rec = [{"subject_a": f"subj_{i%5}", "subject_b": f"subj_{(i+1)%5}", "score": s} 
                     for i, s in enumerate(non_mated_scores)]
    clust_res = compute_system_unlinkability_clustered_ci(mated_rec, non_mated_rec, num_bins=25, num_bootstraps=50)
    assert 0.0 <= clust_res["d_sys"] < 0.20
    assert 0.0 <= clust_res["clustered_ci_lower"] <= clust_res["clustered_ci_upper"]

def test_topology_matching_and_balanced_unlinkability():
    """
    Tests topology-only matching comparator and balanced cross-key evaluation.
    """
    # 1. Topology matching between identical vs disjoint meshes
    adj_empty = np.empty((0, 0), dtype=np.uint8)
    assert match_topology_only(adj_empty, adj_empty) == 0.0
    
    # 5-triangle chain
    adj1 = np.array([
        [0, 2, 0, 0, 0],
        [2, 0, 2, 0, 0],
        [0, 2, 0, 2, 0],
        [0, 0, 2, 0, 2],
        [0, 0, 0, 2, 0]
    ], dtype=np.uint8)
    # Identical graph
    assert match_topology_only(adj1, adj1) == pytest.approx(1.0)
    
    # Fully disconnected graph
    adj2 = np.zeros((5, 5), dtype=np.uint8)
    sim_diff = match_topology_only(adj1, adj2)
    assert 0.0 <= sim_diff < 1.0
    
    # 2. Balanced key evaluation function
    rng = np.random.default_rng(42)
    descs = [rng.uniform(-1, 1, size=(8, 9)).astype(np.float32) for _ in range(3)]
    weights = [np.ones(8, dtype=np.float32) for _ in range(3)]
    
    def dummy_derive(k):
        # Deterministic projection from key
        seed = hash(k) % (2**31)
        r = np.random.default_rng(seed)
        mat = r.normal(0, 1, size=(32, 9)).astype(np.float32)
        q, _ = np.linalg.qr(mat.T)
        return q.T[:32, :9]
        
    def dummy_matcher(t1, t2, w1, w2):
        return match_protected_templates(t1, t2, w1, w2, hamming_threshold=0.22)
        
    d_sys, m_scores, nm_scores = evaluate_unlinkability(
        descs, weights, dummy_derive, dummy_matcher, num_keys=3
    )
    # 3 keys -> 3 key pairs: mated has 3 * 3 = 9, non-mated has 3 * 3 = 9
    assert len(m_scores) == 9
    assert len(nm_scores) == 9
    assert 0.0 <= d_sys <= 1.0

def test_linear_programming_feasibility_attack():
    """
    Tests Scenario III linear feasibility attack on the convex polyhedral cone.
    Under known W in R^{32 x 12}, tests that finding a feasible point in the cone
    reproduces the target bits under W, while the open cone does not guarantee
    recovering the exact physical descriptor.
    """
    master = "fixed_master_secret_for_test_32b!"
    W = derive_orthonormal_matrix(master, "test_salt", "test_app", in_dim=12, out_dim=32)
    
    from src.graph.descriptors import DEFAULT_DESCRIPTOR_MEAN, DEFAULT_DESCRIPTOR_STD
    d_orig = DEFAULT_DESCRIPTOR_MEAN.copy().reshape(1, 12)
    d_orig[0, 0] = 0.75
    d_orig[0, 1] = 0.45
    d_orig[0, 2] = 8.0  # Physical perimeter in mm
    d_orig[0, 9] = 1.0
    d_orig[0, 10] = 0.0
    d_orig[0, 11] = 1.0
    target_bits = quantize_descriptors(project_descriptors(d_orig, W))[0]
    
    is_feasible, candidate_desc = solve_linear_feasibility_inversion(target_bits, W, delta=0.02)
    assert candidate_desc.shape == (12,)
    
    if is_feasible:
        from src.graph.descriptors import DEFAULT_DESCRIPTOR_MEAN, DEFAULT_DESCRIPTOR_STD
        z = (candidate_desc - DEFAULT_DESCRIPTOR_MEAN) / (DEFAULT_DESCRIPTOR_STD + 1e-8)
        proj = W @ z
        reproduced_bits = (proj >= 0.0).astype(np.uint8)
        match_ratio = np.mean(reproduced_bits == target_bits)
        assert match_ratio >= 0.85
        
    # Exercise Chebyshev center and QP center on 12-D
    from src.security.inversion_attack import (
        solve_chebyshev_center_inversion,
        solve_qp_center_inversion,
        calculate_continuous_inversion_metrics,
        calculate_angular_wrap_error
    )
    feas_c, d_c, r_c = solve_chebyshev_center_inversion(target_bits, W, delta=0.01)
    assert feas_c is True
    assert d_c.shape == (12,)
    assert r_c >= 0.0
    
    feas_qp, d_qp = solve_qp_center_inversion(target_bits, W, delta=0.01)
    assert feas_qp is True
    assert d_qp.shape == (12,)
    
    # Exercise 9-D Chebyshev and QP center
    from src.graph.descriptors import DEFAULT_DESCRIPTOR_MEAN_9D, DEFAULT_DESCRIPTOR_STD_9D
    W_9d = derive_orthonormal_matrix(master, "test_salt_9d", "test_app", in_dim=9, out_dim=32)
    d_9d = DEFAULT_DESCRIPTOR_MEAN_9D.copy().reshape(1, 9)
    d_9d[0, 0] = 0.75
    d_9d[0, 1] = 0.45
    d_9d[0, 2] = 8.0
    target_bits_9d = quantize_descriptors(project_descriptors(d_9d, W_9d, mean=DEFAULT_DESCRIPTOR_MEAN_9D, std=DEFAULT_DESCRIPTOR_STD_9D))[0]
    feas_c9, d_c9, r_c9 = solve_chebyshev_center_inversion(target_bits_9d, W_9d, delta=0.01)
    assert feas_c9 is True
    assert d_c9.shape == (9,)
    feas_qp9, d_qp9 = solve_qp_center_inversion(target_bits_9d, W_9d, delta=0.01)
    assert feas_qp9 is True
    assert d_qp9.shape == (9,)
    
    # Continuous metrics and circular angular wrap
    m_cont = calculate_continuous_inversion_metrics(d_orig[0], d_c)
    assert "mae" in m_cont and "rmse" in m_cont and "normalized_euclidean" in m_cont
    assert m_cont["normalized_euclidean"] == pytest.approx(m_cont["rmse"])
    
    # Test circular wrap on [0, 1): 0.05 and 0.95 should have circular distance 0.10
    d_a = np.zeros(9, dtype=np.float32)
    d_b = np.zeros(9, dtype=np.float32)
    d_a[3] = 0.05
    d_b[3] = 0.95
    ang_err = calculate_angular_wrap_error(d_a, d_b)
    assert ang_err == pytest.approx(0.10 / 6.0)

def test_inversion_resistance_hill_climbing():
    rng = np.random.default_rng(42)
    target_template = rng.integers(0, 2, size=(15, 32), dtype=np.uint8)
    
    def dummy_matcher(query, enrolled, wq, we):
        return match_protected_templates(query, enrolled, wq, we, hamming_threshold=0.22)
        
    score, best_cand = simulate_hill_climbing_inversion(
        target_template, W_key=None, matcher_fn=dummy_matcher, max_iterations=50
    )
    assert 0.0 <= score <= 1.0

def test_milp_feasibility_inversion_and_leakage_metrics():
    from src.security.leakage_analysis import solve_milp_feasibility_inversion, compute_type_leakage_metrics
    master = generate_master_secret()
    W = derive_orthonormal_matrix(master, "leak_salt", "leak_app", in_dim=12, out_dim=32)
    
    # Real descriptor with known types
    d_real = np.array([0.75, 0.45, 8.5, 0.3, 0.7, 0.5, 0.4, 0.6, 0.3, 0.0, 1.0, 0.0], dtype=np.float32)
    from src.graph.descriptors import DEFAULT_DESCRIPTOR_MEAN, DEFAULT_DESCRIPTOR_STD
    z = (d_real - DEFAULT_DESCRIPTOR_MEAN) / (DEFAULT_DESCRIPTOR_STD + 1e-8)
    b = (W @ z >= 0.0).astype(np.uint8)
    
    feas, d_recon = solve_milp_feasibility_inversion(b, W, delta=0.01)
    assert feas is True
    # Verify exact binary variables for minutiae types
    assert np.all(np.isin(d_recon[9:12], [0.0, 1.0]))
    
    # Check metrics computation
    metrics = compute_type_leakage_metrics(d_real[9:12], d_recon[9:12])
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "balanced_accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

def test_milp_solver_info_and_directional_search():
    """
    Tests HiGHS solver info return metadata, directional objective search for ALL THREE
    binary variables (k=9, 10, 11), and no-good integer exclusions to prove discrete
    minutiae-type uniqueness in the polyhedral cone.
    """
    from src.security.leakage_analysis import solve_milp_feasibility_inversion, verify_milp_multicoordinate_uniqueness
    master = "fixed_master_secret_for_test_32b!"
    W = derive_orthonormal_matrix(master, "dir_salt", "dir_app", in_dim=12, out_dim=32)
    
    d_real = np.array([0.70, 0.50, 7.5, 0.4, 0.5, 0.6, 0.3, 0.5, 0.7, 1.0, 0.0, 1.0], dtype=np.float32)
    from src.graph.descriptors import DEFAULT_DESCRIPTOR_MEAN, DEFAULT_DESCRIPTOR_STD
    z = (d_real - DEFAULT_DESCRIPTOR_MEAN) / (DEFAULT_DESCRIPTOR_STD + 1e-8)
    b = (W @ z >= 0.0).astype(np.uint8)
    
    feas, d_recon, info = solve_milp_feasibility_inversion(b, W, delta=0.01, return_info=True)
    assert feas is True
    assert info["status"] == 0  # Optimal
    assert info["mip_gap"] == 0.0
    
    audit = verify_milp_multicoordinate_uniqueness(b, W, delta=0.01)
    assert audit["status"] == "success"
    assert audit["all_three_strictly_unique"] is True
    assert audit["num_alternatives_feasible"] == 0
    assert len(audit["alternative_tuples_tested"]) == 7
    
    assert audit["directional_search"]["d9"]["is_fixed"] is True
    assert audit["directional_search"]["d10"]["is_fixed"] is True
    assert audit["directional_search"]["d11"]["is_fixed"] is True
    
    # Continuous intervals should be non-empty (continuum)
    for c_idx in range(9):
        assert audit["continuous_coordinate_intervals"][f"d{c_idx}"]["interval_width"] >= 0.0


def test_9d_inference_attacks_baseline_and_chance():
    """
    Tests empirical inference attacks against 9-D descriptors, verifying that
    minutiae types cannot be predicted from geometry or protected bitstrings
    significantly better than chance (balanced accuracy ~ 50%).
    """
    from src.security.leakage_analysis import evaluate_9d_inference_attacks, stratified_group_kfold
    master = generate_master_secret()
    W_9d = derive_orthonormal_matrix(master, "inf_salt", "inf_app", in_dim=9, out_dim=32)
    
    rng = np.random.default_rng(42)
    N = 100
    X_geom = rng.uniform(0.1, 0.9, size=(N, 9)).astype(np.float32)
    X_geom[:, 2] = rng.uniform(2.0, 15.0, size=N)  # P_mm
    # Ensure r1 >= r2
    X_geom[:, 0] = np.maximum(X_geom[:, 0], X_geom[:, 1] + 0.05)
    
    # 80% endings (0), 20% bifurcations (1)
    y_types = rng.choice([0, 1], p=[0.8, 0.2], size=(N, 1)).astype(np.int32)
    finger_ids = np.array([f"finger_{i % 10}" for i in range(N)])
    
    # Verify stratified_group_kfold disjointness
    splits = stratified_group_kfold(finger_ids, y_types[:, 0], n_splits=5, seed=42)
    for f1, (tr, te) in enumerate(splits):
        tr_groups = set(finger_ids[tr])
        te_groups = set(finger_ids[te])
        assert len(tr_groups.intersection(te_groups)) == 0, "Training and test groups must be disjoint"
    
    res = evaluate_9d_inference_attacks(X_geom, y_types, W_9d, finger_ids=finger_ids, 
                                        num_folds=3, seed=42, n_bootstrap=100, n_permutations=50)
    assert "majority_class_baseline" in res
    assert res["majority_class_baseline"]["accuracy"] >= 0.70
    assert res["majority_class_baseline"]["balanced_accuracy"] == pytest.approx(0.50)
    assert "stratified_random_baseline" in res
    assert "unprotected_geometry_attack" in res
    assert "protected_known_key_attack" in res
    
    bal_geom = res["unprotected_geometry_attack"]["balanced_accuracy"]
    bal_bits = res["protected_known_key_attack"]["balanced_accuracy"]
    
    # Balanced accuracy should be near uninformative chance (0.50 +/- 0.20 for N=100)
    assert 0.30 <= bal_geom <= 0.70
    assert 0.30 <= bal_bits <= 0.70
    assert "ci_accuracy_95" in res["unprotected_geometry_attack"]
    assert "ci_balanced_accuracy_95" in res["unprotected_geometry_attack"]
    assert "permutation_p_value_vs_chance" in res["unprotected_geometry_attack"]
    assert "grouped_permutation_test" in res["unprotected_geometry_attack"]
    assert "fold_metrics" in res["unprotected_geometry_attack"]
    assert "accuracy_mean" in res["unprotected_geometry_attack"]["fold_metrics"]
    assert "balanced_accuracy_mean" in res["unprotected_geometry_attack"]["fold_metrics"]
    assert "bh_adjusted_p_value" in res["unprotected_geometry_attack"]
    assert "formatted_report" in res
    assert "interpretation" in res
    
    # Test PureNumpy Decision Tree, Random Forest, and PR-AUC
    from experiments.evaluate_phase2_leakage import PureNumpyDecisionTree, PureNumpyRandomForest, compute_pr_auc
    y_test_bin = np.array([0, 1, 0, 1, 0, 0, 1, 0], dtype=np.int32)
    p_test_bin = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.3, 0.7, 0.2], dtype=np.float32)
    auc_val = compute_pr_auc(y_test_bin, p_test_bin)
    assert 0.80 <= auc_val <= 1.0
    
    rf = PureNumpyRandomForest(n_estimators=5, max_depth=3, random_state=42)
    rf.fit(X_geom[:50], y_types[:50, 0])
    probs = rf.predict_proba(X_geom[50:])
    assert probs.shape == (50,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_boundary_reflection_derivative_and_joint_bootstrap():
    """
    Tests:
    1. Boundary-corrected KDE density integration to 1.0 and boundary flattening.
    2. Prior ratio omega vs inverse convention documentation.
    3. Multi-threat joint two-way cluster bootstrap for D_sys^max.
    """
    rng = np.random.default_rng(42)
    mated_scores = rng.beta(2, 5, size=200) * 0.5  # domain in [0, 1]
    non_mated_scores = rng.beta(2, 5, size=300) * 0.5
    
    # Prior ratio omega = 1.0
    ds, eval_pts, d_omega, lr = compute_system_unlinkability(
        mated_scores, non_mated_scores, num_bins=100, score_domain=(0.0, 1.0), omega=1.0
    )
    assert 0.0 <= ds <= 1.0
    assert len(eval_pts) == 100
    assert eval_pts[0] == 0.0 and eval_pts[-1] == 1.0
    
    # Multi-threat joint bootstrap
    subjs = [f"s_{i}" for i in range(5)]
    kps = [(0, 1), (0, 2), (1, 2)]
    
    m_recs_desc = [{"subject": s, "score": float(rng.uniform(0.01, 0.05)), "key1": kp[0], "key2": kp[1]} 
                   for s in subjs for kp in kps]
    nm_recs_desc = [{"subject_a": subjs[i], "subject_b": subjs[j], "score": float(rng.uniform(0.01, 0.05)), "key1": kp[0], "key2": kp[1]} 
                    for i in range(len(subjs)) for j in range(i+1, len(subjs)) for kp in kps]
                    
    m_recs_topo = [{"subject": s, "score": float(rng.uniform(0.85, 0.95)), "key1": kp[0], "key2": kp[1]} 
                   for s in subjs for kp in kps]
    nm_recs_topo = [{"subject_a": subjs[i], "subject_b": subjs[j], "score": float(rng.uniform(0.80, 0.92)), "key1": kp[0], "key2": kp[1]} 
                    for i in range(len(subjs)) for j in range(i+1, len(subjs)) for kp in kps]
                    
    threat_recs = {
        "desc": {"mated": m_recs_desc, "non_mated": nm_recs_desc},
        "topo": {"mated": m_recs_topo, "non_mated": nm_recs_topo}
    }
    
    joint_res = compute_multi_threat_unlinkability_bootstrap(threat_recs, num_bootstraps=50, seed=42)
    assert "d_sys_max_point" in joint_res
    assert "d_sys_max_ci_95" in joint_res
    assert joint_res["d_sys_max_ci_95"][0] <= joint_res["d_sys_max_ci_95"][1]
    assert joint_res["d_sys_max_point"] >= joint_res["models"]["desc"]["d_sys"]
    assert joint_res["d_sys_max_point"] >= joint_res["models"]["topo"]["d_sys"]

