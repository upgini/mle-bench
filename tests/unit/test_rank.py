from pathlib import Path

from mlebench.rank import collect_rankings_data


def test_collect_rankings_data_returns_results():
    repo_dir = Path(__file__).resolve().parents[2]
    run_group_experiments = repo_dir / "runs" / "run_group_experiments.csv"
    runs_dir = repo_dir / "runs"
    splits_dir = repo_dir / "experiments" / "splits"
    competition_categories = repo_dir / "experiments" / "competition_categories.csv"
    experiment_agents = repo_dir / "runs" / "experiment_agents.csv"
    sample_report = repo_dir / "runs" / "sample-submissions" / "grading_report.json"

    results = collect_rankings_data(
        run_group_experiments_path=run_group_experiments,
        runs_dir=runs_dir,
        splits_dir=splits_dir,
        competition_categories_path=competition_categories,
        split_type="low",
        competition_category="all",
        experiment_agents_path=experiment_agents,
        sample_report_path=sample_report,
    )

    assert results is not None
    assert len(results.competition_stats) > 0
    assert set(results.competition_stats.keys()).issubset(set(results.competitions))
    assert not results.final_results.empty
