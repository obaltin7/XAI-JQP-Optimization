"""
run_xai_on_mistral.select_passages: the default behavior (first N eligible paragraphs) is unchanged;
--passage-offset yields a second, disjoint calibration sample (--seed does not change the passages).

Run:
    pytest tests/test_select_passages_mini.py -q
"""
import importlib.machinery
import importlib.util
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _name, _attrs in (("captum", {}), ("captum.attr", {"LayerIntegratedGradients": None}), ("datasets", {"load_dataset": None})):
    if importlib.util.find_spec(_name.split(".")[0]) is None and _name not in sys.modules:  # local .venv: no datasets/captum
        _m = types.ModuleType(_name)
        _m.__spec__ = importlib.machinery.ModuleSpec(_name, None)  # so find_spec calls in other tests do not raise ValueError
        for k, v in _attrs.items():
            setattr(_m, k, v)
        sys.modules[_name] = _m
        if "." in _name:
            setattr(sys.modules[_name.split(".")[0]], _name.split(".")[1], _m)

from run_xai_on_mistral import parse_args, select_passages  # noqa: E402

TEXTS = [" = Heading = ", "", "a" * 250, "short", "b" * 300, " = = Sub = = ", "c" * 260, "d" * 270, "e" * 280]


def test_default_is_first_n_and_offset_skips_eligible_only():
    assert select_passages(TEXTS, 2, 200) == ["a" * 250, "b" * 300]
    assert select_passages(TEXTS, 2, 200, offset=0) == select_passages(TEXTS, 2, 200)
    assert select_passages(TEXTS, 2, 200, offset=2) == ["c" * 260, "d" * 270]  # headings/short lines do not count towards the offset
    assert set(select_passages(TEXTS, 2, 200)).isdisjoint(select_passages(TEXTS, 2, 200, offset=2))
    assert select_passages(TEXTS, 3, 200, offset=4) == ["e" * 280]  # too few: warning, no error
    try:
        select_passages(TEXTS, 1, 200, offset=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("a negative offset must raise ValueError")


def test_cli_defaults_unchanged():
    a = parse_args([])
    assert a.passage_offset == 0 and a.seed is None and a.n_passages == 16 and a.n_steps == 8
    assert parse_args(["--passage-offset", "16", "--seed", "1"]).passage_offset == 16
