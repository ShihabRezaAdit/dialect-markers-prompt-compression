"""Minimal test runner (no pytest needed): runs every test_* function in tests/test_*.py.
Exit code 0 only if all pass."""
import importlib.util
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))


def main() -> int:
    failed = passed = 0
    for path in sorted(HERE.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            try:
                getattr(mod, name)()
                passed += 1
            except Exception:
                failed += 1
                print(f"FAIL {path.name}::{name}")
                traceback.print_exc()
    print(f"{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
