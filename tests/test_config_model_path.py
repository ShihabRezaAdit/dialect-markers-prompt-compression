"""model_path uses the pinned snapshot folder in the cache and never contacts the Hub offline."""
import os
import tempfile

from idea86.config import model_path


def test_model_path_reads_cached_snapshot_offline():
    old = {k: os.environ.get(k) for k in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE")}
    try:
        with tempfile.TemporaryDirectory() as d:
            os.environ.pop("HF_HUB_CACHE", None)
            os.environ["HF_HOME"], os.environ["HF_HUB_OFFLINE"] = d, "1"
            snap = os.path.join(d, "hub", "models--org--m", "snapshots", "abc123")
            os.makedirs(snap)
            open(os.path.join(snap, "config.json"), "w").write("{}")
            assert model_path("org/m", "abc123") == snap
            try:
                model_path("org/m", "other")
                raise AssertionError("missing revision not reported")
            except SystemExit as e:
                assert "not in the cache" in str(e)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
