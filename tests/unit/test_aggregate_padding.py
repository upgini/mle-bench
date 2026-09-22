import json
from pathlib import Path

from experiments.aggregate_grading_reports import aggregate_any_medal


def _write_report(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps({"competition_reports": rows}))


def _row(competition_id: str, medal: bool) -> dict:
    return {
        "competition_id": competition_id,
        "gold_medal": medal,
        "silver_medal": False,
        "bronze_medal": False,
        "above_median": medal,
        "submission_exists": True,
        "valid_submission": True,
    }


def test_incomplete_run_groups_are_padded_with_failing_scores(tmp_path: Path):
    complete = tmp_path / "group1.json"
    incomplete = tmp_path / "group2.json"
    _write_report(complete, [_row("comp-a", True), _row("comp-b", True)])
    _write_report(incomplete, [_row("comp-a", False)])

    metrics, padded = aggregate_any_medal([str(complete), str(incomplete)], ["comp-a", "comp-b"])

    assert padded
    assert metrics.metrics["any_medal_percentage"].mean == 50
    assert metrics.num_seeds_averaged == 2


def test_repeated_rows_in_one_file_stay_unpadded_seeds(tmp_path: Path):
    report = tmp_path / "group.json"
    _write_report(
        report,
        [
            _row("comp-a", True),
            _row("comp-b", False),
            _row("comp-a", False),
            _row("comp-b", False),
        ],
    )

    metrics, padded = aggregate_any_medal([str(report)], ["comp-a", "comp-b"])

    assert not padded
    assert metrics.metrics["any_medal_percentage"].mean == 25
    assert metrics.num_seeds_averaged == 2
