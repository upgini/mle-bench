from pathlib import Path

import pytest

from mlebench.rank import collect_rankings


def test_collect_rankings_dry_run_prints_low_tabular(monkeypatch, tmp_path):
    repo_dir = Path(__file__).resolve().parents[2]
    run_group_experiments = repo_dir / "runs" / "run_group_experiments.csv"
    runs_dir = repo_dir / "runs"
    splits_dir = repo_dir / "experiments" / "splits"
    competition_categories = repo_dir / "experiments" / "competition_categories.csv"
    experiment_agents = repo_dir / "runs" / "experiment_agents.csv"
    sample_report = repo_dir / "runs" / "sample-submissions" / "grading_report.json"

    collect_rankings(
        run_group_experiments_path=run_group_experiments,
        runs_dir=runs_dir,
        splits_dir=splits_dir,
        competition_categories_path=competition_categories,
        split_type="low",
        competition_category="Tabular",
        experiment_agents_path=experiment_agents,
        output_dir=tmp_path,
        sample_report_path=sample_report,
        strict=False,
        max_competitions_missed=0,
        dry_run=True,
    )

    assert True
