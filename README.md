# MLE-bench tabular

[![Open in HF Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/Upgini/mle-bench-tabular)

This is a fork of [MLE-bench](https://github.com/openai/mle-bench) that compares agent performance on tabular data. It uses exactly the same setup and differs just in the leaderboard view. We focus on tabular tasks and use normalized score instead of medal percentage to compare differently scaled scores. The leaderboard is [recomputed](#ranking-across-competition-categories) upon updating submitted runs from OpenAI repo.

### Tabular Leaderbord (Low Split)

The table below summarizes the tabular competition rankings for the Low complexity split. 

| Agent | LLM(s) used | Date | [Normalized Score](#mean-normalized-score) | Any Medal (%) |
| --- | --- | --- | --- | --- |
| [FM Agent](https://github.com/baidubce/FM-Agent) | Gemini-2.5-Pro | 2025-10-10 | 0.944 ± 0.103 | 50.00 ± 0.00 |
| [Upgini](https://github.com/upgini/upgini) + [MLZero](https://github.com/upgini/autogluon-assistant) [^3] | o3-mini | 2025-11-14 | 0.927 ± 0.086 | 50.00 ± 0.00 |
| [MLZero](https://github.com/autogluon/autogluon-assistant) | o3-mini | 2025-11-14 | 0.926 ± 0.088 | 50.00 ± 0.00 |
| [CAIR](https://research.google/teams/cloud-ai-research/) MLE-STAR-Pro-1.5  | Gemini-2.5-Pro | 2025-11-25 | 0.903 ± 0.130 | 50.00 ± 0.00 |
| [Thesis](https://thesislabs.ai) | gpt-5-codex | 2025-11-10 | 0.891 ± 0.150 | 50.00 ± 0.00 |
| [AIDE](https://github.com/wecoai/aideml) | claude-3-5-sonnet-20240620 | 2024-10-08 | 0.874 ± 0.142 | 41.67 ± 8.33 |
| [R&D-Agent](https://github.com/microsoft/RD-Agent) | o1-preview | 2025-05-14 | 0.818 ± 0.306 | 50.00 ± 0.00 |
| [AIDE](https://github.com/wecoai/aideml) | gpt-4o-2024-08-06 | 2024-10-08 | 0.808 ± 0.136 | 36.84 ± 2.79 |
| [R&D-Agent](https://github.com/microsoft/RD-Agent) | o3 + GPT-4.1 | 2025-08-15 | 0.793 ± 0.371 | 50.00 ± 0.00 |
| [AIDE](https://github.com/wecoai/aideml) | o1-preview | 2024-10-08 | 0.783 ± 0.421 | 40.00 ± 3.80 |
| [Operand](https://operand.com) ensemble | gpt-5 (low verbosity/effort) | 2025-10-06 | 0.780 ± 0.282 | 50.00 ± 0.00[^2] |
| [CAIR](https://research.google/teams/cloud-ai-research/) MLE-STAR-Pro | Gemini-2.5-Pro | 2025-11-03 | 0.727 ± 0.532 | 50.00 ± 0.00 |
| [Neo](https://heyneo.so/) multi-agent | undisclosed | 2025-07-28 | 0.723 ± 0.483 | 50.00 ± 0.00 |
| [InternAgent](https://github.com/Alpha-Innovator/InternAgent/) | deepseek-r1 | 2025-09-12 | 0.711 ± 0.518 | 50.00 ± 0.00 |
| [ML-Master](https://github.com/zeroxleo/ML-Master) | deepseek-r1 | 2025-06-17 | 0.687 ± 0.600 | 41.67 ± 8.33 |
| [R&D-Agent](https://github.com/microsoft/RD-Agent) | gpt-5 | 2025-09-26 | 0.497 ± 0.574 | 50.00 ± 0.00 |
| [Leeroo](https://leeroo.com/) | Gemini-3-Pro-Preview[^4] | 2025-12-07 | 0.495 ± 0.572 |  50.00 ± 0.00[^2] |
| OpenHands | gpt-4o-2024-08-06 | 2024-10-08 | 0.342 ± 0.605 | 41.67 ± 8.33 |
| [AIDE](https://github.com/wecoai/aideml) | llama-3.1-405b-instruct | 2024-10-08 | 0.328 ± 1.032 | 35.00 ± 10.00 |
| MLAB | gpt-4o-2024-08-06 | 2024-10-08 | -0.110 ± 0.392 | 15.63 ± 4.57 |

[^1]: With some light assistance from an ensemble of models including
    Gemini-2.5-Pro, Grok-4, and Claude 4.1 Opus, distilled by Gemini-2.5-Pro.
[^2]: Computed by padding incomplete seeds with failing scores.
[^3]: A fork with added integration with Upgini in the data processing step
[^4]: The architecture is primarily driven by Gemini-3-Pro-Preview, with a subset of modules utilizing GPT-5 and GPT-5-mini.

### Mean Normalized Score

The score above is the mean across scores normalized between sample submission score and the gold medal score: 

The formula for normalized score is:

```
normalized score = (sample submission score - score) / (sample submission score - gold medal score)
```

where:
- **score** is the agent's raw score on a particular competition.
- **gold medal score** is the score achieved by the gold medal baseline.
- **sample submission score** is the score achieved by the public sample submission or baseline.

This normalization ensures scores are comparable across competitions with different scales. A value of `1` thus means best human result.

### Low Split Overall Leaderboard

The table below shows the overall Low split leaderboard for all competition categories. It is ranked by normalized score, with any medal score added for reference. 

| Agent | LLM(s) used | Date | Normalized Score | Any Medal (%) |
| --- | --- | --- | --- | --- |
| [CAIR](https://research.google/teams/cloud-ai-research/) MLE-STAR-Pro-1.5  | Gemini-2.5-Pro | 2025-11-25 | 0.940 ± 0.149 | 68.18 ± 2.62 |
| [FM Agent](https://github.com/baidubce/FM-Agent) | Gemini-2.5-Pro | 2025-10-10 | 0.909 ± 0.201 | 62.12 ± 1.52 |
| [InternAgent](https://github.com/Alpha-Innovator/InternAgent/) | deepseek-r1 | 2025-09-12 | 0.893 ± 0.264 | 62.12 ± 3.03 |
| [Thesis](https://thesislabs.ai) | gpt-5-codex | 2025-11-10 | 0.886 ± 0.218 | 65.15 ± 1.52 |
| [Operand](https://operand.com) ensemble | gpt-5 (low verbosity/effort) | 2025-10-06 | 0.883 ± 0.194 | 63.64 ± 0.00 |
| [R&D-Agent](https://github.com/microsoft/RD-Agent) | o1-preview | 2025-05-14 | 0.880 ± 0.199 | 48.18 ± 1.11 |
| [ML-Master](https://github.com/zeroxleo/ML-Master) | deepseek-r1 | 2025-06-17 | 0.864 ± 0.311 | 48.48 ± 1.52 |
| [AIDE](https://github.com/wecoai/aideml) | o1-preview | 2024-10-08 | 0.856 ± 0.236 | 35.91 ± 1.86 |
| [R&D-Agent](https://github.com/microsoft/RD-Agent) | o3 + GPT-4.1 | 2025-08-15 | 0.837 ± 0.321 | 51.52 ± 4.01 |
| [CAIR](https://research.google/teams/cloud-ai-research/) MLE-STAR-Pro | Gemini-2.5-Pro | 2025-11-03 | 0.822 ± 0.411 | 66.67 ± 1.52 |
| [R&D-Agent](https://github.com/microsoft/RD-Agent) | gpt-5 | 2025-09-26 | 0.746 ± 0.428 | 68.18 ± 2.62 |
| [Leeroo](https://leeroo.com/) | Gemini-3-Pro-Preview[^4] | 2025-12-07 | 0.716 ± 0.452 |  68.18 ± 2.62[^2] |
| OpenHands | gpt-4o-2024-08-06 | 2024-10-08 | 0.342 ± 0.605 | 41.67 ± 8.33 |
| [Neo](https://heyneo.so/) multi-agent | undisclosed | 2025-07-28 | 0.699 ± 0.382 | 48.48 ± 1.52 |
| [AIDE](https://github.com/wecoai/aideml) | gpt-4o-2024-08-06 | 2024-10-08 | 0.661 ± 0.334 | 18.55 ± 1.26 |
| [AIDE](https://github.com/wecoai/aideml) | claude-3-5-sonnet-20240620 | 2024-10-08 | 0.505 ± 0.584 | 19.70 ± 1.52 |
| OpenHands | gpt-4o-2024-08-06 | 2024-10-08 | 0.430 ± 0.392 | 12.12 ± 1.52 |
| MLAB | gpt-4o-2024-08-06 | 2024-10-08 | 0.299 ± 0.426 | 4.55 ± 0.86 |
| [AIDE](https://github.com/wecoai/aideml) | llama-3.1-405b-instruct | 2024-10-08 | 0.276 ± 0.541 | 10.23 ± 1.14 |


### Producing Medal Scores for the Leaderboard

To produce the scores for the leaderboard, please organize your grading reports
in the `runs/` folder organized by run groups, with one grading report per run
group. Identify the run groups for your submission in
`runs/run_group_experiments.csv` with an experiment id. Then run

```bash
uv run python experiments/aggregate_grading_reports.py --experiment-id <exp_id> --split low
uv run python experiments/aggregate_grading_reports.py --experiment-id <exp_id> --split medium
uv run python experiments/aggregate_grading_reports.py --experiment-id <exp_id> --split high
uv run python experiments/aggregate_grading_reports.py --experiment-id <exp_id> --split split75
```

Report the mean and standard error of the mean (SEM) for each of the splits on
the reported `any_medal_percentage` metric. The `--split75` flag corresponds to
the `All (%)` column.

## Benchmarking

This section describes a canonical setup for comparing scores on MLE-bench. We recommend the following:
- Repeat each evaluation with at least 3 seeds and report the Any Medal (%) score as the mean ± one standard error of the mean. The evaluation (task and grading) itself is deterministic, but agents/LLMs can be quite high-variance!
- Agent resources - not a strict requirement of the benchmark but please report if you stray from these defaults!
  - Runtime: 24 hours
  - Compute: 36 vCPUs with 440GB RAM and one 24GB A10 GPU
- Include a breakdown of your scores across Low, Medium, High, and All complexity [splits](experiments/splits) (see *Lite evaluation* below for why this is useful).

### Lite Evaluation

Evaluating agents with the above settings on the full 75 competitions of MLE-bench can be expensive. For users preferring a "lite" version of the benchmark, we recommend using the [Low complexity split](https://github.com/openai/mle-bench/blob/main/experiments/splits/low.txt) of our dataset, which consists of only 22 competitions. This reduces the number of runs substantially, while still allowing fair comparison along one column of the table above.

Furthermore, the Low complexity competitions tend to be significantly more lightweight (158GB total dataset size compared to 3.3TB for the full set), so users may additionally consider reducing the runtime or compute resources available to the agents for further cost reduction. However, note that doing so risks degrading the performance of your agent. For example, see [Section 3.3 and 3.4 of our paper](https://arxiv.org/abs/2410.07095) where we have experimented with varying resources on the full competition set.

The Lite dataset contains the following competitions:

| Competition ID                              | Category                   | Dataset Size (GB) |
|---------------------------------------------|----------------------------|--------------------|
| aerial-cactus-identification                | Image Classification       | 0.0254            |
| aptos2019-blindness-detection               | Image Classification       | 10.22             |
| denoising-dirty-documents                   | Image To Image             | 0.06              |
| detecting-insults-in-social-commentary      | Text Classification        | 0.002             |
| dog-breed-identification                    | Image Classification       | 0.75              |
| dogs-vs-cats-redux-kernels-edition          | Image Classification       | 0.85              |
| histopathologic-cancer-detection            | Image Regression           | 7.76              |
| jigsaw-toxic-comment-classification-challenge | Text Classification        | 0.06              |
| leaf-classification                         | Image Classification       | 0.036             |
| mlsp-2013-birds                             | Audio Classification       | 0.5851            |
| new-york-city-taxi-fare-prediction          | Tabular                   | 5.7               |
| nomad2018-predict-transparent-conductors    | Tabular                   | 0.00624           |
| plant-pathology-2020-fgvc7                  | Image Classification       | 0.8               |
| random-acts-of-pizza                        | Text Classification        | 0.003             |
| ranzcr-clip-catheter-line-classification    | Image Classification       | 13.13             |
| siim-isic-melanoma-classification           | Image Classification       | 116.16            |
| spooky-author-identification                | Text Classification        | 0.0019            |
| tabular-playground-series-dec-2021          | Tabular                   | 0.7               |
| tabular-playground-series-may-2022          | Tabular                   | 0.57              |
| text-normalization-challenge-english-language | Seq->Seq                 | 0.01              |
| text-normalization-challenge-russian-language | Seq->Seq                 | 0.01              |
| the-icml-2013-whale-challenge-right-whale-redux | Audio Classification     | 0.29314           |

## Setup

Some MLE-bench competition data is stored using [Git-LFS](https://git-lfs.com/).
Once you have downloaded and installed LFS, run:

```console
git lfs fetch --all
git lfs pull
```

You can install `mlebench` with pip:

```console
pip install -e .
```

### Pre-Commit Hooks (Optional)

If you're committing code, you can install the pre-commit hooks by running:

```console
pre-commit install
```

## Dataset

The MLE-bench dataset is a collection of 75 Kaggle competitions which we use to evaluate the ML engineering capabilities of AI systems.

Since Kaggle does not provide the held-out test set for each competition, we
provide preparation scripts that split the publicly available training set into
a new training and test set.

For each competition, we also provide grading scripts that can be used to
evaluate the score of a submission.

We use the [Kaggle API](https://github.com/Kaggle/kaggle-api) to download the
raw datasets. Ensure that you have downloaded your Kaggle credentials
(`kaggle.json`) and placed it in the `~/.kaggle/` directory (this is the default
location where the Kaggle API looks for your credentials). To download and prepare the MLE-bench dataset, run the following, which will download and prepare the dataset in your system's default cache directory. Note, we've found this to take two days when running from scratch:

```console
mlebench prepare --all
```

To prepare the lite dataset, run:

```console
mlebench prepare --lite
```

Alternatively, you can prepare the dataset for a specific competition by
running:

```console
mlebench prepare -c <competition-id>
```

Run `mlebench prepare --help` to see the list of available competitions.



## Grading Submissions

Answers for competitions must be submitted in CSV format; the required format is described in each competition's description, or shown in a competition's sample submission file. You can grade multiple submissions by using the `mlebench grade` command. Given a JSONL file, where each line corresponds with a submission for one competition, `mlebench grade` will produce a grading report for each competition. The JSONL file must contain the following fields:
- `competition_id`: the ID of the competition in our dataset.
- `submission_path`: a `.csv` file with the predictions for the specified
  competition.

See more information by running `mlebench grade --help`.

You can also grade individual submissions using the `mlebench grade-sample` command. For example, to grade a submission for the Spaceship Titanic competition, you can run:

```console
mlebench grade-sample <PATH_TO_SUBMISSION> spaceship-titanic
```

See more information by running `mlebench grade-sample --help`.

## Ranking across competition categories

It's possible to rank existing results for a particular split and competition category. For this, you can run:

```console
mlebench rank  --split-type <split type> --competition-category <category>
```

By default, `mlebench rank` will calculate rankings for Low Tabular split.

You can also exclude from calculations those agents that don't have enough competition entries, e.g.:
```console
mlebench rank --competition-category all --strict --max-competitions-missed 10
```
This is the command used to calculate overall leaderboard for Low split above.

This command saves normalized scores for each competition plus overall ranking in separate files. See more information by running `mlebench rank --help`.

## Environment

We provide a base Docker image `mlebench-env` which is the base environment for our agents. This base image contains:
- Conda environment used to execute our agents. We optionally (default true) install Python packages in this environment which are commonly used across our agents. If you don't want to install these packages, set the `INSTALL_HEAVY_DEPENDENCIES` environment variable to `false` when building the image, by adding `--build-arg INSTALL_HEAVY_DEPENDENCIES=false` to the `docker build` command below
- Instructions for agents to follow when creating their submission
- Grading server for agents to use when checking that the structure of their submission is correct

Build this image by running:

```bash
docker build --platform=linux/amd64 -t mlebench-env -f environment/Dockerfile .
```

## Agents

We purposefully designed our benchmark to not make any assumptions about the agent that produces submissions, so agents can more easily be evaluated on this benchmark. We evaluated three open-source agents; we discuss this procedure in [agents/README.md](agents/README.md).

## Extras

We include additional features in the MLE-bench repository that may be useful
for MLE-bench evaluation. These include a rule violation detector and
a plagiarism detector. We refer readers to
[extras/README.md](extras/README.md) for more information.

## Examples

We collect example usage of this library in the `examples/` directory, see [examples/README.md](examples/README.md) for more information.

## Experiments

We place the code specific to the experiments from our publication of the
benchmark in the `experiments/` directory:
- For instance, our competition splits are available in `experiments/splits/`.
- For a completed set of runs from a given agent, you can use the provided
`experiments/make_submission.py` script to compile its submission for grading.
- We release our methodology for the "familiarity" experiments in `experiments/familiarity/`, see [experiments/familiarity/README.md](experiments/familiarity/README.md) for more information.

## Dev

Note, when running `pytest` locally, be sure to accept the competition rules otherwise the tests will fail.

## Known Issues

There are some known issues with certain MLE-bench competitions. Since we have
already received leaderboard submissions, we are postponing fixes to avoid
invalidating the leaderboard. Instead, we plan to release batched fixes in the
upcoming v2 release of MLE-bench on the
[openai/frontier-evals](https://github.com/openai/frontier-evals) repo, which will
include a version column in the leaderboard to distinguish between v1 and v2 results.
If you wish to make a submission to v1 in the meantime, please still include
the following competitions in your overall scores. The known issues are
catalogued below:

- **tensorflow-speech-recognition-challenge**:
  - The prepare.py script incorrectly prepares the test set such that there is a
    much larger range of test labels than there should be.
    [#63](https://github.com/openai/mle-bench/issues/63)
  - The prepare.py script does not properly create a test set where the speaker
    IDs are disjoint from those in train/val.
- **icecube-neutrinos-in-deep-ice**: Checksums are mismatch.
  [#58](https://github.com/openai/mle-bench/issues/58)
- **ranzcr-clip-catheter-line-classification**: The prepare.py script results in
  missing columns in the sample submission.
  [#30](https://github.com/openai/mle-bench/issues/30)
- **tabular-playground-series-dec-2021**: The leaderboard is crowded -- very
  little difference between the top score and the median score.
- **tabular-playground-series-may-2022**: The leaderboard is crowded -- very
  little difference between the top score and the median score.
- **jigsaw-toxic-comment-classification-challenge**: The leaderboard is crowded -- very
  little difference between the top score and the median score.
- **champs-scalar-coupling**: test molecules are missing in structures.csv.
  [#70](https://github.com/openai/mle-bench/pull/70)
- **multi-modal-gesture-recognition**: public test `.mat` files leak test labels.
  [#77](https://github.com/openai/mle-bench/issues/77)
- **smartphone-decimeter-2022**: The public test `span_log.nmea` files leak
  information that makes achieving a perfect score trivial.
  [#93](https://github.com/openai/mle-bench/issues/93)
- **hubmap-kidney-segmentation**: The public test `{image_id}.json` files leak
  information that makes achieving a close-to-perfect score trivial. They should
  be removed.

## Authors

Chan Jun Shern, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, Lilian Weng, Aleksander Mądry

## Citation

Please cite using the following BibTeX entry:
```
@article{chan2024mle-bench,
  title={MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering},
  author={Jun Shern Chan and Neil Chowdhury and Oliver Jaffe and James Aung and Dane Sherburn and Evan Mays and Giulio Starace and Kevin Liu and Leon Maksin and Tejal Patwardhan and Lilian Weng and Aleksander Mądry},
  year={2024},
  eprint={2410.07095},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2410.07095}
}
```
