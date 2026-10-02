"""pytest bootstrap: expose the repository root, src/, experiments/ and tests/ on sys.path.

The project keeps flat module names (``import compressor``, ``from run_ablation_tests import ...``)
so that the same import statements work both on the GPU host and in the local test-suite.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src"), os.path.join(_REPO_ROOT, "experiments"), os.path.join(_REPO_ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
