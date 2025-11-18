# MLE-bench tabular

This is a fork of [MLE-bench](https://github.com/openai/mle-bench) that compares agent performance on tabular data. It uses exactly the same setup and differs just in the leaderboard view. We focus on tabular tasks and use normalized score instead of medal percentage to compare differently scaled scores. The leaderboard is recomputed upon updating submitted runs from OpenAI repo.

### Tabular Leaderbord (Lite Split)

The table below summarizes the tabular competition rankings for the Lite complexity split. 

| Agent | LLM(s) used | [Normalized Score](#mean-normalized-score) (Mean ± Std) | Any Medal (Mean ± Std) |
| --- | --- | --- | --- |
| [FM Agent](https://github.com/baidubce/FM-Agent) | Gemini-2.5-Pro | 0.944 ± 0.103 | 0.500 ± 0.577 |
| [Upgini](https://github.com/upgini/upgini) + [MLZero](https://github.com/upgini/autogluon-assistant) [^3] | o3-mini | 0.927 ± 0.086 | 0.500 ± 0.577 |
| [MLZero](https://github.com/autogluon/autogluon-assistant) | o3-mini | 0.926 ± 0.088 | 0.500 ± 0.577 |
| [Thesis](https://thesislabs.ai) | gpt-5-codex | 0.891 ± 0.150 | 0.500 ± 0.577 |
| AIDE | claude-3-5-sonnet-20240620 | 0.874 ± 0.142 | 0.500 ± 0.577 |
| AIDE | gpt-4o-2024-08-06 | 0.857 ± 0.145 | 0.375 ± 0.479 |
| [R&D-Agent](https://github.com/microsoft/RD-Agent) | o1-preview | 0.818 ± 0.306 | 0.500 ± 0.577 |
| [R&D-Agent](https://github.com/microsoft/RD-Agent) | o3 + GPT-4.1 | 0.793 ± 0.371 | 0.500 ± 0.577 |
| AIDE | o1-preview | 0.783 ± 0.421 | 0.500 ± 0.577 |
| [Operand](https://operand.com) ensemble | gpt-5 (low verbosity/effort) | 0.780 ± 0.282 | 0.500 ± 0.577 |
| [Neo](https://heyneo.so/) multi-agent | undisclosed | 0.723 ± 0.483 | 0.500 ± 0.577 |
| [R&D-Agent](https://github.com/microsoft/RD-Agent) | gpt-5 | 0.497 ± 0.574 | 0.500 ± 0.577 |
| [InternAgent](https://github.com/Alpha-Innovator/InternAgent/) | deepseek-r1 | 0.048 ± 1.841 | 0.500 ± 0.577 |
| AIDE | llama-3.1-405b-instruct | 0.041 ± 1.603 | 0.250 ± 0.500 |
| [ML-Master](https://github.com/zeroxleo/ML-Master) | deepseek-r1 | -10.396 ± 22.766 | 0.417 ± 0.500 |
| [CAIR](https://research.google/teams/cloud-ai-research/) MLE-STAR-Pro | Gemini-2.5-Pro | -12.560 ± 27.105 | 0.500 ± 0.577 |
| OpenHands | gpt-4o-2024-08-06 | -17.743 ± 36.238 | 0.375 ± 0.479 |
| MLAB | gpt-4o-2024-08-06 | -1083553262524.917 ± 2167106525049.096 | 0.083 ± 0.167 |

[^3]: A fork with added integration with Upgini in the data processing step

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

This saves normalized scores for each competition plus overall ranking in separate files. See more information by running `mlebench rank --help`.

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
