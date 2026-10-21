"""Build the demo index (run once, and again after adding new sites).

    python demo/build_index.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    n = core.build_index()
    print(f"done in {time.time() - t0:.0f} s")
