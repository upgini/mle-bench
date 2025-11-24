import json
import re
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import List

import pandas as pd

from mlebench.utils import get_logger, read_csv

logger = get_logger(__name__)


@dataclass
class RankingResults:
    competitions: list[str]
    competition_stats: dict[str, pd.DataFrame]
    final_results: pd.DataFrame


def _safe_path_component(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^\w\-\.]", "-", cleaned)
    return cleaned.lower()


def load_competitions(
    split_type: str,
    competition_category: str,
    splits_dir: Path,
    competition_categories_path: Path,
    logger: Logger,
) -> List[str]:
    """Load competition IDs filtered by split and, optionally, competition category."""
    split_file = splits_dir / f"{split_type}.txt"

    if not split_file.exists():
        logger.error(f"Split file not found for split '{split_type}': {split_file}")
        return []

    split_competitions = [
        line.strip()
        for line in split_file.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if len(split_competitions) == 0:
        logger.warning(f"No competitions listed in split file: {split_file}")

    normalized_category = competition_category.strip().casefold()
    include_all_categories = normalized_category == "all"

    if include_all_categories:
        logger.info(
            f"Loaded {len(split_competitions)} competitions for split '{split_type}' across all categories."
        )
        return split_competitions

    competition_categories = read_csv(competition_categories_path)

    if competition_categories.empty:
        logger.error(f"Competition categories file is empty: {competition_categories_path}")
        return []

    competition_categories["category"] = competition_categories["category"].astype(str).str.strip()
    category_filter = (
        competition_categories["category"].str.casefold() == competition_category.strip().casefold()
    )
    category_competitions = set(
        competition_categories.loc[category_filter, "competition_id"].dropna().astype(str)
    )

    if len(category_competitions) == 0:
        logger.error(
            f"No competitions found with category '{competition_category}' in {competition_categories_path}"
        )
        return []

    competitions = [
        competition_id
        for competition_id in split_competitions
        if competition_id in category_competitions
    ]

    if len(competitions) == 0:
        logger.warning(
            f"No competitions from split '{split_type}' match category '{competition_category}'. "
            "Check split and category configuration."
        )

    logger.info(
        f"Loaded {len(competitions)} competitions for split '{split_type}' and category '{competition_category}'."
    )

    return competitions


def get_competition_results(
    competition_id: str, runs_dir: Path, experiment_groups: pd.DataFrame, logger: Logger
) -> pd.DataFrame | None:
    """Collect all results for a specific competition across all run groups."""
    results = []

    # Get unique run groups from experiment_groups
    run_groups = experiment_groups["run_group"].unique()

    for run_group in run_groups:
        run_group_dir = runs_dir / run_group
        if not run_group_dir.exists() or not run_group_dir.is_dir():
            continue

        # Find grading report files
        grading_reports = sorted(list(run_group_dir.glob("*grading_report*.json")))
        if len(grading_reports) == 0:
            continue

        # Load the latest grading report
        latest_report = grading_reports[-1]
        data = json.loads(latest_report.read_text())

        reports = data.get("competition_reports", [])
        if isinstance(reports, dict):
            reports = [reports]

        for report in reports:
            if report.get("competition_id") == competition_id:
                report["run_group"] = run_group
                results.append(report)
                break

    if len(results) == 0:
        logger.warning(f"No results found for competition: {competition_id}")
        return None

    results_df = pd.DataFrame(results)

    # Merge with experiment groups
    results_df = results_df.merge(experiment_groups, how="left", on="run_group")

    return results_df


def get_any_medal_results(
    runs_dir: Path, experiment_groups: pd.DataFrame, competition_ids: List[str], logger: Logger
) -> pd.DataFrame:
    results = []

    # Get unique run groups from experiment_groups
    run_groups = experiment_groups["run_group"].unique()

    for run_group in run_groups:
        run_group_dir = runs_dir / run_group
        if not run_group_dir.exists() or not run_group_dir.is_dir():
            continue

        # Find grading report files
        grading_reports = sorted(list(run_group_dir.glob("*grading_report*.json")))
        if len(grading_reports) == 0:
            continue

        # Load the latest grading report
        latest_report = grading_reports[-1]

        data = json.loads(latest_report.read_text())
        reports = data.get("competition_reports", [])
        if isinstance(reports, dict):
            reports = [reports]
        reports_df = pd.DataFrame(
            [
                report
                for report in reports
                if report.get("competition_id") in competition_ids
                and report.get("valid_submission")
            ]
        )
        if not reports_df.empty:
            medals = reports_df.groupby("competition_id").agg(any_medal=("any_medal", "median"))
            medal_pct = medals.reindex(competition_ids).fillna(0).mean(axis=None)
            results.append({"run_group": run_group, "medal_pct": medal_pct})
        else:
            results.append({"run_group": run_group, "medal_pct": 0.0})

    if len(results) == 0:
        logger.warning("No any medal results found")
        return pd.DataFrame(columns=["run_group", "medal_pct"])

    results_df = pd.DataFrame(results)

    # Merge with experiment groups
    results_df = experiment_groups.merge(results_df, how="left", on="run_group")
    results_mean = results_df.groupby("experiment_id")["medal_pct"].mean().rename("mean_medal_pct")
    results_sem = (
        results_df.groupby("experiment_id")["medal_pct"].sem(ddof=2).rename("sem_medal_pct")
    )
    results_df = pd.concat([results_mean, results_sem], axis=1)
    results_df = results_df.reset_index()
    return results_df


def load_sample_reports(sample_report_path: Path, logger: Logger) -> dict[str, dict]:
    sample_reports: dict[str, dict] = {}
    if not sample_report_path.exists():
        logger.warning(f"Sample grading report not found at {sample_report_path}")
        return sample_reports
    try:
        sample_data = json.loads(sample_report_path.read_text())
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse sample grading report at {sample_report_path}: {exc}")
        return sample_reports

    sample_competitions = sample_data.get("competition_reports", [])
    if isinstance(sample_competitions, dict):
        sample_competitions = [sample_competitions]

    for report in sample_competitions:
        competition_id = report.get("competition_id")
        if competition_id:
            sample_reports[competition_id] = report
    return sample_reports


def score_competition_results(
    competition_id: str,
    results_df: pd.DataFrame,
    sample_reports: dict[str, dict],
    logger: Logger,
) -> pd.DataFrame | None:
    sample_report = sample_reports.get(competition_id)
    if not sample_report:
        logger.warning(
            "Sample submission score not found for %s; skipping competition in aggregated rankings.",
            competition_id,
        )
        return None

    sample_score = sample_report.get("score")
    sample_gold = sample_report.get("gold_threshold")
    if sample_score is None or sample_gold is None:
        logger.warning(
            "Sample submission score or gold threshold missing for %s; skipping competition in aggregated rankings.",
            competition_id,
        )
        return None

    denominator: float | None = None
    denominator = sample_score - sample_gold
    if denominator == 0:
        logger.warning(
            "Sample submission score equals gold threshold for %s; normalized scores unavailable.",
            competition_id,
        )
        return None

    results_df["normalized_score"] = (sample_score - results_df["score"]) / denominator

    stats_df = results_df.groupby("experiment_id", group_keys=True).agg(
        mean_score=("score", "mean"),
        std_score=("score", "std"),
        n_runs=("score", "count"),
        mean_normalized_score=("normalized_score", "mean"),
        std_normalized_score=("normalized_score", "std"),
    )
    stats_df = stats_df.reset_index()

    stats_df = stats_df.sort_values(
        "mean_normalized_score", ascending=False, na_position="last"
    ).reset_index(drop=True)

    return stats_df


def aggregate_scores(rank_series_list: list[pd.Series], name: str) -> pd.DataFrame | None:
    if len(rank_series_list) == 0:
        return None
    rank_df = pd.concat(rank_series_list, axis=1).fillna(0)
    rank_df = rank_df.agg(["mean", "std"], axis=1)
    rank_df.columns = [f"mean_{name}", f"std_{name}"]
    return rank_df


def collect_rankings_data(
    run_group_experiments_path: Path,
    runs_dir: Path,
    splits_dir: Path,
    competition_categories_path: Path,
    split_type: str,
    competition_category: str,
    experiment_agents_path: Path,
    sample_report_path: Path,
    strict: bool = False,
    max_competitions_missed: int = 0,
) -> RankingResults | None:
    sample_reports = load_sample_reports(sample_report_path, logger)

    # Load competitions from configured split and category
    competitions = load_competitions(
        split_type=split_type,
        competition_category=competition_category,
        splits_dir=splits_dir,
        competition_categories_path=competition_categories_path,
        logger=logger,
    )

    if len(competitions) == 0:
        logger.error("No competitions available to process. Exiting.")
        return
    logger.info(f"Competitions to evaluate: {competitions}")

    # Read experiment to run_group mapping
    experiment_groups = read_csv(run_group_experiments_path)
    logger.info(f"Found {len(experiment_groups)} experiment-run_group mappings")

    # Read experiment agents mapping
    experiment_agents = read_csv(experiment_agents_path)
    logger.info(f"Found {len(experiment_agents)} experiment-agent mappings")

    # Collect results for each tabular competition
    rank_series_list: list[pd.Series] = []
    competition_stats: dict[str, pd.DataFrame] = {}

    for competition_id in competitions:
        logger.info(f"Processing competition: {competition_id}")
        results_df = get_competition_results(competition_id, runs_dir, experiment_groups, logger)

        if results_df is None or len(results_df) == 0:
            logger.warning(f"No results for {competition_id}")
            continue

        # Filter to only valid scores
        results_df = results_df[results_df["score"].notna()].copy()

        if len(results_df) == 0:
            logger.warning(f"No valid scores for {competition_id}")
            continue

        stats_df = score_competition_results(competition_id, results_df, sample_reports, logger)

        if stats_df is None:
            continue

        competition_stats[competition_id] = stats_df

        rank_series_list.append(
            stats_df.set_index("experiment_id")["mean_normalized_score"].rename(competition_id)
        )

    if len(rank_series_list) == 0:
        logger.error("No scores collected for any competition!")
        return

    # Create overall score DataFrame
    per_competition_rank_df = pd.concat(rank_series_list, axis=1)
    rank_df = aggregate_scores(rank_series_list, "normalized_score")

    if strict:
        rank_df = exclude_agents_with_missing_results(
            per_competition_rank_df, rank_df, max_competitions_missed
        )
        if len(rank_df) == 0:
            logger.error(
                "Strict ranking removed all experiments; cannot compute overall ranking. "
                "Ensure every agent has results for all competitions with sample scores."
            )
            return
    elif max_competitions_missed > 0:
        logger.info(
            "max_competitions_missed=%d specified but strict ranking disabled; parameter has no effect.",
            max_competitions_missed,
        )

    medal_df = get_any_medal_results(runs_dir, experiment_groups, competitions, logger)

    # Create final results DataFrame
    final_results = rank_df.merge(medal_df, on="experiment_id", how="left").reset_index(drop=True)
    final_results = final_results.merge(experiment_agents, on="experiment_id", how="left")

    # Sort by mean_rank
    final_results = final_results.sort_values("mean_normalized_score", ascending=False).reset_index(
        drop=True
    )

    return RankingResults(
        competitions=competitions,
        competition_stats=competition_stats,
        final_results=final_results,
    )


def save_rankings_results(
    ranking_results: RankingResults,
    output_dir: Path,
    split_type: str,
    competition_category: str,
    dry_run: bool = False,
):
    split_dirname = _safe_path_component(split_type)
    category_dirname = _safe_path_component(competition_category)
    competition_results_dir = output_dir / split_dirname / category_dirname / "competition_results"
    base_output_dir = competition_results_dir.parent

    if dry_run:
        logger.info(
            "Dry run enabled; skipping file writes. Final rankings will be printed to stdout."
        )
        for competition_id in ranking_results.competition_stats:
            logger.info("Dry run: skipping write for competition %s", competition_id)
        logger.info(
            ranking_results.final_results[
                ["experiment_id", "mean_normalized_score", "mean_medal_pct", "sem_medal_pct"]
            ]
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    competition_results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing results under {base_output_dir}")

    for competition_id, stats_df in ranking_results.competition_stats.items():
        competition_output = competition_results_dir / f"{competition_id}.csv"
        stats_df.to_csv(competition_output, index=False)
        logger.info(f"Saved results for {competition_id} to {competition_output}")

    final_output = base_output_dir / "overall_ranks.csv"
    ranking_results.final_results.to_csv(final_output, index=False)
    logger.info(f"Saved final ranking to {final_output}")


def collect_rankings(
    run_group_experiments_path: Path,
    runs_dir: Path,
    splits_dir: Path,
    competition_categories_path: Path,
    split_type: str,
    competition_category: str,
    experiment_agents_path: Path,
    output_dir: Path,
    sample_report_path: Path,
    strict: bool = False,
    max_competitions_missed: int = 0,
    dry_run: bool = False,
):
    ranking_results = collect_rankings_data(
        run_group_experiments_path=run_group_experiments_path,
        runs_dir=runs_dir,
        splits_dir=splits_dir,
        competition_categories_path=competition_categories_path,
        split_type=split_type,
        competition_category=competition_category,
        experiment_agents_path=experiment_agents_path,
        sample_report_path=sample_report_path,
        strict=strict,
        max_competitions_missed=max_competitions_missed,
    )
    if ranking_results is None:
        return
    save_rankings_results(
        ranking_results=ranking_results,
        output_dir=output_dir,
        split_type=split_type,
        competition_category=competition_category,
        dry_run=dry_run,
    )


def exclude_agents_with_missing_results(
    per_competition_rank_df: pd.DataFrame,
    rank_df: pd.DataFrame,
    max_competitions_missed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_competitions_missed < 0:
        logger.warning(
            "max_competitions_missed (%d) is negative; treating it as 0.",
            max_competitions_missed,
        )
        max_competitions_missed = 0

    required_competitions = per_competition_rank_df.shape[1]
    coverage_counts = per_competition_rank_df.notna().sum(axis=1)
    missing_counts = required_competitions - coverage_counts
    allowed_mask = missing_counts <= max_competitions_missed
    permitted_experiments = missing_counts[allowed_mask].index
    removed_experiments = set(rank_df.index) - set(permitted_experiments)
    if len(removed_experiments) > 0:
        logger.info(
            (
                "Strict ranking enabled; excluding %d experiment(s) missing more than %d competition result(s) "
                "out of %d competitions: %s"
            ),
            len(removed_experiments),
            max_competitions_missed,
            required_competitions,
            ", ".join(sorted(removed_experiments)),
        )
    rank_df = rank_df.loc[permitted_experiments]

    return rank_df
