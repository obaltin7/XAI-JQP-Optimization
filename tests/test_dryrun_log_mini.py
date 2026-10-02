"""
--dry-run / --preflight never write to the tracked default logs (log_gun6.txt etc.) or to the real results/*.json.

Regression: `run_ablation_tests.py --dry-run --preflight` used to append lines to the tracked log_gun6.txt. Rule
(run_e2e_pipeline.should_log_to_file): dry-run/preflight + DEFAULT log path -> console only; an explicit --log or the
dry-run redirection to dryrun_out/ -> written to file; a normal run -> always written (behavior unchanged).

Run:
    pytest tests/test_dryrun_log_mini.py -q
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_baselines, run_ablation_tests, run_iterative_pruning
from run_e2e_pipeline import Logger, should_log_to_file  # noqa: E402


def _ns(**kw):
    return argparse.Namespace(**{"dry_run": False, "preflight": False, "log": "log_x.txt", **kw})


def test_should_log_to_file_rule():
    assert should_log_to_file(_ns(), "log_x.txt")  # normal run: always written
    assert not should_log_to_file(_ns(dry_run=True), "log_x.txt")
    assert not should_log_to_file(_ns(preflight=True), "log_x.txt")
    assert should_log_to_file(_ns(dry_run=True, log="dryrun_out/log.txt"), "log_x.txt")  # redirected / explicit --log
    assert should_log_to_file(argparse.Namespace(log="log_x.txt"), "log_x.txt")  # script without dry_run/preflight flags


def test_logger_to_file_false_prints_but_does_not_create_file(tmp_path, capsys):
    path = tmp_path / "log.txt"
    Logger(str(path), to_file=False)("hello", tag="T")
    assert "[T] hello" in capsys.readouterr().out and not path.exists()
    Logger(str(path))("hello", tag="T")  # default: original behavior
    assert "[T] hello" in path.read_text(encoding="utf-8")


def test_gun6_dry_run_preflight_leaves_tracked_log_and_results_untouched(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the default relative paths (log_gun6.txt, results/, dryrun_out/) would be created here
    assert run_ablation_tests.main(["--dry-run", "--preflight"]) == 0
    assert not (tmp_path / run_ablation_tests.LOG_FILE).exists() and not (tmp_path / "results").exists()
    assert run_ablation_tests.main(["--dry-run", "--configs", "prune_only", "--skip-baseline"]) == 0  # full dry run
    assert not (tmp_path / run_ablation_tests.LOG_FILE).exists() and not (tmp_path / "results").exists()
    assert (tmp_path / "dryrun_out" / "ablation_gun6_dry.json").exists()
    explicit = tmp_path / "explicit_log.txt"  # written when an explicit --log is given
    assert run_ablation_tests.main(["--dry-run", "--preflight", "--log", str(explicit)]) == 0
    assert "Preflight completed" in explicit.read_text(encoding="utf-8")


def test_gun7_dry_run_preflight_does_not_touch_default_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run_iterative_pruning.main(["--dry-run", "--preflight", "--fractions", "0.2", "--configs", "xai_single"]) == 0
    assert not (tmp_path / run_iterative_pruning.LOG_FILE).exists()  # the dry-run log goes to dryrun_out/
    assert (tmp_path / "dryrun_out" / "log_gun7_dry.txt").exists()


def test_default_outputs_unchanged_and_dry_run_outputs_redirected():
    for mod in (run_ablation_tests, run_baselines):
        assert mod.parse_args([]).output == mod.OUTPUT_FILE and mod.parse_args([]).log == mod.LOG_FILE
        dry = mod.parse_args(["--dry-run"])
        assert dry.output.startswith("dryrun_out") and dry.output != mod.OUTPUT_FILE
        assert mod.parse_args(["--dry-run", "--output", "x.json"]).output == "x.json"
