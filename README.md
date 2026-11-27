# Benchmark analyzer

`ba` compares the five open-weight models that stand highest on the
[Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard), using
only what the leaderboard publishes: each model's transcripts of its eight public English
sets. It reproduces the leaderboard's own WER from those files, then asks more of them: CER
and semantic error, six normalizers, what each normalization rule is worth to each model,
and breakdowns by what is known about the test cases. A small web UI shows it all.

Nothing is transcribed here. Every number is a score of a transcript the leaderboard
published, so no key and no account is needed anywhere, and anyone can redo all of it.

## Install

```bash
git clone --recurse-submodules git@github.com:rjtrout/benchmark-analyzer.git
cd benchmark-analyzer
uv sync
uv run ba reproduce
uv sync --extra demo
uv run ba ui
```

A run of all the fetching and scoring has been saved to Github under artifacts because the time to compute everything is extensive. A fresh download will take a while, but be saved on some hours of computation. To start fresh, this is the order after the clone above and sync. This will take considerable time for the download and the semantic embedding


## Starting fresh
```
uv run ba clear all
uv run ba fetch --audio      # starts artifacts/ over, and measures the metadata as it reads
uv run ba reproduce
uv run ba score --metric wer --metric cer
uv run ba ablate
uv run ba ablate --metric cer
uv run ba ablate --normalizer leaderboard_split
uv run ba ablate --normalizer leaderboard_split --metric cer
uv run ba semantic --families
uv run ba ui
```

Python 3.11 or later and [`uv`](https://docs.astral.sh/uv/). The Open ASR Leaderboard code
and the URGENT authors' WADA-SNR implementation are Git submodules under `third_party/`.

## The comparison

**Models**, in `configs/systems.yaml`
**Benchmarks**, in `configs/benchmarks.yaml`

## Fetching: `ba fetch`

```bash
uv run ba fetch 
uv run ba fetch --audio
```

It reads two things, both public, with no account:

- **Each benchmark's test case list** from its dataset: the test case IDs, the references
  and the metadata the dataset gives. No audio. Every model and every normalizer is scored
  on that one list.
- **Each model's transcripts** from the leaderboard's bucket
  (`hf-audio/asr_leaderboard_h200`)

If run with `--audio` it will download audio as well (used for full metadata).  This is recommended

A fetch starts over by removing everything stored
under `artifacts/`. 

A fetch takes several minutes and downloads about 200 MB of text into`data/`, which git ignores.

## Reproducing the leaderboard's numbers

```bash
uv run ba reproduce
```

It scores every file again under the leaderboard's rules and puts each WER beside the two
figures the leaderboard publishes: the score file beside the transcript file, and the
table. It exits with 1 when a published WER is not the one the public transcripts give.

## Scoring

A row is stored for every test case, not an average, so every figure is a query over them.

- **Normalizers.** `verbatim` (none), `case-punct` (the floor: case and punctuation only,
  with an apostrophe inside a word kept as spelling), `whisper-stock` (OpenAI
  Whisper's, as shipped), `leaderboard` (the leaderboard's own, unchanged),
  `leaderboard_split` (the leaderboard's rules as 13 families that switch off: with every
  family on it equals `leaderboard` exactly) and `own`: those 13 families and four more of
  ours.
- **Metrics.** `wer`, `cer` and `semantic`.
- **Aggregations.** `mean` and `median` over test cases, and `overall`: all errors over all
  reference words.
- **Earnings22 is scored per call**, as the leaderboard does: a call's chunks are joined
  before they are compared with the call's one reference.

```bash
uv run ba score --normalizer leaderboard --metric wer --metric cer
uv run ba score --normalizer own --without numerals --benchmark voxpopuli-cleaned
uv run ba ablate --benchmark urgent-noisy             # what each rule family of `own` is worth
uv run ba ablate --normalizer leaderboard_split       # the same for the leaderboard's rules alone
uv run ba slice --by snr_given --benchmark urgent-noisy        # a score against a field, in 10 bins
uv run ba slice --by ref_words --bins 4 --benchmark ami-cleaned
```

`ba score --help` names the 17 rule families of `own`, and `ba slice --help` the ten fields.

### Slices

`ba slice --by <field>` and the Slices page break a score down by one of ten fields:
`duration`, `sampling_rate`, `level`, `snr_given`, `snr_est`, `bandwidth_est`,
`speech_fraction`, `ref_words`, `speaking_rate` and `alphanumeric`. A number is cut into 10
bins that each hold the same number of test cases, and `--bins` changes that. A number with
fewer values than bins, such as a sampling rate, gets a bin for each value. A flag gives
false and true.
Every row shows the test cases and reference words behind its score, and test cases with no
value for the field are counted together in a last row.

### What each rule family is worth

`ba ablate`, and the heatmaps on the Normalization page, score a model once with every rule
family of a normalizer on and once with each family off. Two normalizers are made of
families.

WER and CER are counted as the table is asked for. Semantic error is read: `ba semantic
--families` computes it for each family switched off. This takes ~38 minutes for the 30 variants of the two normalizers.

### Clearing saved results and downloads

Everything worked out here is stored under `artifacts/` and read back: the texts as each
normalizer leaves them, and the scores per test case. The store is a snapshot.

```bash
uv run ba clear scores             # the normalized texts and scores computed here
uv run ba clear leaderboard-scores # the imported leaderboard score Parquet tables
uv run ba clear test-cases         # the saved test case tables, including metadata
uv run ba clear audio              # the downloaded audio
uv run ba clear transcript-files   # the downloaded transcript files
```

To refresh one normalizer and keep the rest, score it again from the transcripts as written:

```bash
uv run ba score --rescore --normalizer own --metric wer --metric cer
uv run ba semantic --recompute --normalizer own
```

### Semantic error
It runs on the local machine with one sentence encoder, `sentence-transformers/all-mpnet-base-v2`.
Its library is installed with everything else, and its weights are downloaded the first
time it is used. Semantic error is stored beside WER and CER. Compute it from the Results
page, or with:

```bash
uv run ba semantic
uv run ba score --metric semantic --normalizer leaderboard
```

The models on a benchmark share its references, so those are embedded once. All five
models, seven sets and the first five normalizers took about an hour on a laptop.

It is not given for `earnings22-cleaned`: a whole call is far more text than a sentence encoder reads.

## The web UI

```bash
uv sync --extra demo
uv run ba ui
```
## Metadata

Ten fields describe a test case apart from any model's transcript of it. Each is a column
of the benchmark's test cases.

| Field | Where it comes from |
| --- | --- |
| `duration` | Given: the length the dataset states, taken as it is |
| `snr_given` | Given: the level the noise was mixed at, which URGENT's noisy half states |
| `sampling_rate` | Given: the audio file's own rate |
| `ref_words`, `alphanumeric` | Derived from the reference text: its words, and whether it holds a number |
| `speaking_rate` | Derived from those words and the length: words a second, pauses included |
| `snr_est` | Derived from the audio using the URGENT authors' WADA-SNR implementation |
| `level`, `bandwidth_est`, `speech_fraction` | Derived from the audio: loudness, bandwidth and the share of speech |

`ba fetch` fills them in as it reads a benchmark. The sampling rate and the
four fields measured on the audio need the audio, so run fetch on the audio.

```bash
uv run ba fetch --audio                                   # 20 GB for all eight sets
uv run ba fetch --audio --benchmark urgent-noisy          # or one set: 0.4 GB
uv run ba metadata                                        # derive again, from the references and the audio on disk
```

`ba metadata` leaves what was measured on the audio as it is where the audio is not on disk.
The Hugging Face library keeps its own copy of what it downloads, so the fetch takes about
twice that on disk until its cache is cleared.

`ba test_metadata` tests the fields against what else is known of the same audio: the noise
level URGENT states and its clean half, by what URGENT's authors tagged as done to each
recording, and a sample of clips changed in a known way. It reads the audio on disk, prints
figures and saves nothing. It is not part of the test suite.

```bash
uv run ba test_metadata
```

## Commands

```
ba fetch [--system ID] [--benchmark ID] [--audio]
ba reproduce
ba score [--system ID] [--benchmark ID] [--normalizer N] [--metric M] [--agg A] [--without FAMILY] [--rescore]
ba ablate [--system ID] [--benchmark ID] [--normalizer N] [--metric M] [--agg A] [--recompute]
ba semantic [--system ID] [--benchmark ID] [--normalizer N] [--families] [--recompute]
ba metadata [--benchmark ID]
ba test_metadata
ba slice --by F [--system ID] [--benchmark ID] [--metric M] [--bins N] [--agg A]
ba clear scores|leaderboard-scores|test-cases|audio|transcript-files
ba ui
```

## Tests

```bash
uv run pytest -m "not golden"      # the code, on small made-up data. No network
uv run pytest -m golden            # the reproduction, on the stored transcripts
uv run ruff check src tests
```

## Layout

```
configs/          the models and the benchmarks
artifacts/        test cases, references, transcripts and scores: Parquet tables, no audio, about 820 MB
data/             downloads as fetched, the audio among them: local cache, git-ignored
third_party/      the Open ASR Leaderboard and URGENT analysis code, as submodules
src/ba/
  paths.py        where things live
  config.py       the models and the benchmarks, read from configs/
  store/          the data on disk, under artifacts/: a table to a module
    store.py           Store: the five tables behind one interface, and what clears them
    test_cases.py      a benchmark's test cases
    transcriptions.py  references and transcripts, as written and as each normalizer leaves them
    scores.py          our scores, a metric at a time, and the leaderboard's own
    parquet.py         how a file is written and read
  cli/            the commands: one module for each, named after it, put together in main.py
  fetching/       everything `ba fetch` reads from outside: the datasets and the public bucket
  actions/        what the commands and pages do with the stored data
    normalization.py  the texts as a normalizer leaves them: normalized once, and kept
    scoring.py        a run's scores per test case
    score_summary.py  the score table: a figure for each run, normalizer and metric, with its counts
    ablation.py       what each rule family is worth
    reproduce.py      ours beside the leaderboard's published figures
    slices.py         a score broken down by a metadata field, and the test cases behind a bin
    test_metadata.py  tests of the metadata fields, run on the real project by `ba test_metadata`
  metrics/        wer.py, cer.py, semantic.py, and registry.py: the table of metrics
  aggregations/   aggregations.py: mean, median and overall, and summarize. registry.py: the table of them
  normalize/      normalizer.py, one module for each normalizer, and registry.py: the table of them
  metadata/       the ten fields: given.py and derived.py, and populate.py, which fills both in
                  acoustic.py: the measures taken on the audio
  ui/             the web UI: app.py, a page each under pages/, and what the pages share
  utils/          display_utils.py, audio_utils.py, list_utils.py, registry.py
tests/            one file for each module or package above
```

Each package's `__init__.py` is empty, so a name is always imported from the module that
defines it. Only `ba fetch` downloads anything, and all of that is in `fetching/`: nothing
else in the package imports it. The rest works on what is stored under `artifacts/`:

```
test_cases/<benchmark>.parquet
    a benchmark's test cases, in order: each one's ten metadata fields, and no text
transcriptions/references/<benchmark>/clean.parquet
    the benchmark's references, as written
transcriptions/references/<benchmark>/<normalizer>.parquet
    the same references as a normalizer leaves them
transcriptions/hypotheses/<model>__<benchmark>/clean.parquet
    a model's transcripts, as written
transcriptions/hypotheses/<model>__<benchmark>/<normalizer>.parquet
    the same transcripts as a normalizer leaves them
scores/ours/<model>__<benchmark>/<normalizer>.parquet
    a row per transcript: its word errors, character errors and semantic error
scores/huggingface/<model>__<benchmark>/leaderboard.parquet
    the leaderboard's own score of the model's transcript file
```
