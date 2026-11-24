import numpy as np

from idea86 import stats as st


def test_holm_step_down():
    out = st.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    assert out["a"]["reject"] and not out["c"]["reject"] and not out["b"]["reject"]
    assert abs(out["a"]["p_holm"] - 0.04) < 1e-12
    out2 = st.holm({"a": 0.001, "b": 0.01, "c": 0.02})
    assert all(v["reject"] for v in out2.values())


def test_bootstrap_deterministic_and_ci():
    v = np.random.default_rng(1).normal(0.1, 0.2, 300)
    e1, d1 = st.bootstrap(st.mean_stat(v), len(v), B=500)
    e2, d2 = st.bootstrap(st.mean_stat(v), len(v), B=500)
    assert e1 == e2 and np.array_equal(d1, d2)
    lo, hi = st.percentile_ci(d1, 0.95)
    assert lo < e1 < hi
    assert st.bootstrap_p(d1) < 0.01


def test_item_mean_stat_weights_items():
    s = st.item_mean_stat(np.array([2.0, 0.0]), np.array([2, 2]))
    assert s(np.array([0, 1])) == 0.5 and s(np.array([0, 0])) == 1.0


def test_signal_rule():
    d = np.random.default_rng(2).normal(0.06, 0.01, 2000)
    assert st.signal(0.06, d, 0.04, 1 - 0.05 / 6)["met"]
    assert not st.signal(0.03, d - 0.03, 0.04, 1 - 0.05 / 6)["met"]


def test_tost_wilson_kappa():
    ok, _ = st.tost_equivalent(np.random.default_rng(3).normal(0, 0.005, 2000), 0.03)
    assert ok
    lo, hi = st.wilson(43, 50)
    assert 0.73 < lo < 0.75 and 0.92 < hi < 0.94
    assert st.cohen_kappa(["Y", "N", "Y", "Y"], ["Y", "N", "Y", "Y"]) == 1.0
