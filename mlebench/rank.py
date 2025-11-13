import json
import re
from logging import Logger
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from mlebench.utils import get_logger, read_csv

logger = get_logger(__name__)


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
    """Load competition IDs filtered by split and competition category."""
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
) -> Tuple[pd.DataFrame | None, bool]:
    """Collect all results for a specific competition across all run groups."""
    results = []
    is_lower_better: bool | None = None

    for run_name in runs_dir.iterdir():
        if not run_name.is_dir():
            continue

        # Find grading report files
        grading_reports = sorted(list(run_name.glob("*grading_report*.json")))
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
                report["run_group"] = run_name.name
                if is_lower_better is None and report.get("is_lower_better") is not None:
                    is_lower_better = bool(report.get("is_lower_better"))
                results.append(report)
                break

    if len(results) == 0:
        logger.warning(f"No results found for competition: {competition_id}")
        return None, False

    results_df = pd.DataFrame(results)

    if "is_lower_better" in results_df.columns:
        is_lower_values = results_df["is_lower_better"].dropna()
        if not is_lower_values.empty:
            is_lower_better = bool(is_lower_values.iloc[0])

    if is_lower_better is None:
        is_lower_better = False

    # Merge with experiment groups
    results_df = results_df.merge(experiment_groups, how="left", on="run_group")

    return results_df, is_lower_better


def collect_rankings(
    run_group_experiments_path: Path,
    runs_dir: Path,
    splits_dir: Path,
    competition_categories_path: Path,
    split_type: str,
    competition_category: str,
    experiment_agents_path: Path,
    output_dir: Path,
):
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

    output_dir.mkdir(parents=True, exist_ok=True)

    split_dirname = _safe_path_component(split_type)
    category_dirname = _safe_path_component(competition_category)
    competition_results_dir = output_dir / split_dirname / category_dirname / "competition_results"
    competition_results_dir.mkdir(parents=True, exist_ok=True)
    base_output_dir = competition_results_dir.parent

    logger.info(f"Writing results under {base_output_dir}")

    # Read experiment to run_group mapping
    experiment_groups = read_csv(run_group_experiments_path)
    logger.info(f"Found {len(experiment_groups)} experiment-run_group mappings")

    # Read experiment agents mapping
    experiment_agents = read_csv(experiment_agents_path)
    logger.info(f"Found {len(experiment_agents)} experiment-agent mappings")

    # Collect results for each tabular competition
    rank_series_list: list[pd.Series] = []

    for competition_id in competitions:
        logger.info(f"Processing competition: {competition_id}")
        results_df, is_lower_better = get_competition_results(
            competition_id, runs_dir, experiment_groups, logger
        )

        if results_df is None or len(results_df) == 0:
            logger.warning(f"No results for {competition_id}")
            continue

        # Filter to only valid scores
        results_df = results_df[results_df["score"].notna()].copy()

        if len(results_df) == 0:
            logger.warning(f"No valid scores for {competition_id}")
            continue

        stats_df = (
            results_df.groupby("experiment_id", group_keys=True)["score"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        stats_df.columns = ["experiment_id", "mean_score", "std_score", "n_runs"]
        stats_df["rank"] = stats_df["mean_score"].rank(
            method="min", ascending=not is_lower_better, na_option="keep", pct=True
        )
        stats_df = stats_df.sort_values("rank", ascending=False).reset_index(drop=True)

        # Save per-competition CSV (without agent descriptions)
        competition_output = competition_results_dir / f"{competition_id}.csv"
        stats_df.to_csv(competition_output, index=False)
        logger.info(f"Saved results for {competition_id} to {competition_output}")

        # Collect ranks for overall ranking
        competition_ranks = stats_df.set_index("experiment_id")["rank"].rename(competition_id)
        rank_series_list.append(competition_ranks)

    # Create overall score DataFrame
    if len(rank_series_list) == 0:
        logger.error("No scores collected for any competition!")
        return

    rank_df = pd.concat(rank_series_list, axis=1)
    rank_df = rank_df.mean(axis=1, skipna=False).rename("mean_rank")

    # Create final results DataFrame
    final_results = rank_df.to_frame().reset_index()
    final_results = final_results.merge(experiment_agents, on="experiment_id", how="left")

    # Sort by mean_rank
    final_results = final_results.sort_values("mean_rank", ascending=False).reset_index(drop=True)

    # Save final mean ranks CSV
    final_output = base_output_dir / "overall_ranks.csv"
    final_results.to_csv(final_output, index=False)
    logger.info(f"Saved final mean ranks to {final_output}")
