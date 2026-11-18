import numpy as np
import pytest
from src.evaluation.metrics import (
    compute_roc_metrics, 
    bootstrap_confidence_intervals, 
    compute_eer,
    clopper_pearson_ci,
    clustered_bootstrap_confidence_intervals
)
from src.evaluation.protocols import generate_pair_lists, load_pair_lists

def test_roc_metrics_computation():
    rng = np.random.default_rng(42)
    # Well-separated genuine and impostor distributions
    genuine = rng.normal(0.8, 0.05, size=100)
    impostor = rng.normal(0.1, 0.05, size=200)
    
    m = compute_roc_metrics(genuine, impostor)
    assert m["auc"] > 0.99
    assert m["eer"] < 0.02
    assert m["tar_at_1pct_far"] > 0.95
    
    # Overlapping distributions (EER around 50%)
    genuine_overlap = rng.normal(0.5, 0.1, size=100)
    impostor_overlap = rng.normal(0.5, 0.1, size=100)
    m_overlap = compute_roc_metrics(genuine_overlap, impostor_overlap)
    assert 0.40 <= m_overlap["eer"] <= 0.60

def test_bootstrap_confidence_intervals():
    rng = np.random.default_rng(42)
    genuine = rng.normal(0.7, 0.1, size=50)
    impostor = rng.normal(0.2, 0.1, size=50)
    
    ci = bootstrap_confidence_intervals(genuine, impostor, num_bootstraps=50)
    assert "eer_mean" in ci
    assert "eer_ci" in ci
    assert ci["eer_ci"][0] <= ci["eer_ci"][1]
    assert ci["auc_ci"][0] <= ci["auc_ci"][1]

def test_pair_list_generation(tmp_path):
    finger_ids = ["f1", "f2", "f3", "f4"]
    impressions_per_finger = 3
    
    gen_pairs, imp_pairs = generate_pair_lists(finger_ids, impressions_per_finger, output_dir=str(tmp_path))
    
    # Genuine pairs per finger: 3 choose 2 = 3. For 4 fingers: 4 * 3 = 12 pairs.
    assert len(gen_pairs) == 12
    # Impostor pairs: 4 choose 2 = 6 pairs.
    assert len(imp_pairs) == 6
    
    # Verify load_pair_lists
    loaded_gen, loaded_imp = load_pair_lists(output_dir=str(tmp_path))
    assert len(loaded_gen) == 12
    assert len(loaded_imp) == 6

def test_clopper_pearson_exact_confidence_intervals():
    # 2 false accepts in 300 impostor trials
    low, high = clopper_pearson_ci(2, 300, alpha=0.05)
    assert 0.0008 <= low <= 0.0009
    assert 0.0238 <= high <= 0.0239
    
    # 0 false rejections in 150 trials (Rule of Three upper bound ~2.00%)
    low0, high0 = clopper_pearson_ci(0, 150, alpha=0.05)
    assert low0 == 0.0
    assert 0.024 <= high0 <= 0.025

def test_clustered_bootstrap_confidence_intervals():
    gen_records = [
        {"subject_a": f"f_{i}", "subject_b": f"f_{i}", "score": 0.55 + 0.01 * (i % 5)}
        for i in range(20)
    ]
    imp_records = [
        {"subject_a": f"f_{i}", "subject_b": f"f_{j}", "score": 0.08 + 0.01 * ((i + j) % 5)}
        for i in range(10) for j in range(i + 1, 10)
    ]
    res = clustered_bootstrap_confidence_intervals(gen_records, imp_records, operating_threshold=0.20, num_bootstraps=50)
    assert "clustered_far_ci" in res
    assert "clustered_frr_ci" in res
    assert "eer_ci" in res
    assert "auc_ci" in res
    assert res["clustered_far_ci"][0] <= res["clustered_far_ci"][1]
    assert res["clustered_frr_ci"][0] <= res["clustered_frr_ci"][1]
    assert res["eer_ci"][0] <= res["eer_ci"][1]
    assert res["auc_ci"][0] <= res["auc_ci"][1]

def test_score_inversion_diagnostic_warning():
    """Verifies that an inverted score distribution triggers a UserWarning."""
    # Impostors have higher scores than genuines (inverted)
    gen = np.array([0.1, 0.2, 0.15])
    imp = np.array([0.8, 0.9, 0.85])
    with pytest.warns(UserWarning, match="Score inversion diagnostic triggered"):
        compute_roc_metrics(gen, imp)


def test_paired_clustered_bootstrap_delta():
    from src.evaluation.metrics import paired_clustered_bootstrap_delta
    # Protected system has slightly lower EER than unprotected
    gen_prot = [{"subject_a": f"f_{i%6}", "subject_b": f"f_{i%6}", "score": 0.65 + 0.01 * (i % 5)} for i in range(30)]
    imp_prot = [{"subject_a": f"f_{i}", "subject_b": f"f_{j}", "score": 0.15 + 0.01 * ((i + j) % 5)} for i in range(6) for j in range(i + 1, 6)]
    
    gen_unprot = [{"subject_a": f"f_{i%6}", "subject_b": f"f_{i%6}", "score": 0.60 + 0.01 * (i % 5)} for i in range(30)]
    imp_unprot = [{"subject_a": f"f_{i}", "subject_b": f"f_{j}", "score": 0.20 + 0.01 * ((i + j) % 5)} for i in range(6) for j in range(i + 1, 6)]
    
    res = paired_clustered_bootstrap_delta(gen_prot, imp_prot, gen_unprot, imp_unprot, num_bootstraps=50)
    assert "delta_eer_point" in res
    assert "delta_eer_95ci" in res
    assert "delta_auc_point" in res
    assert "delta_auc_95ci" in res
    assert res["delta_eer_95ci"][0] <= res["delta_eer_95ci"][1]
    assert res["delta_auc_95ci"][0] <= res["delta_auc_95ci"][1]
