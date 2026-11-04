# OpenArena VLM Capstone

## Repository structure

```text
captone_vlm/
├── controlled/
│   ├── final_experiment.py
│   ├── llm_decision.py
│   ├── game_state.py
│   └── temporal_memory.py
├── live/
│   ├── live_demo.py
│   ├── qwen_agent.py
│   ├── controller.py
│   ├── capture.py
│   └── config.py
├── analysis/
│   ├── make_capstone_figures.py
│   ├── make_controlled_summary_table.py
│   └── make_live_latency_figure.py
├── data/
│   ├── controlled/
│   │   └── final_experiment.jsonl
│   └── live/
│       └── 20261102_181322/
│           └── live.jsonl
├── requirements.txt
├── .gitignore
└── README.md
```

## Study components

### Controlled decision-layer experiment

The controlled experiment compares three input modes using `llama3.2:3b` through Ollama:

- `current_only`
- `raw_history`
- `structured_temporal`

The final benchmark contains 10 predefined scenarios with 10 repeated generations per scenario and configuration (300 responses total). The experiment version is `final_temporal_comparison_v1`.

Run from the repository root:

```bash
cd controlled
python final_experiment.py --help
```

Ollama must be running locally and `llama3.2:3b` must be available.

### Live OpenArena prototype

The live prototype uses `qwen3-vl:2b-instruct` through Ollama. Each cycle captures one gameplay screenshot, attempts to pause OpenArena using Escape, performs VLM inference, and executes a finite deterministic action. A motion-based supervisory mechanism can substitute `ESCAPE` following low visual displacement.

The retained evaluation log in this repository is run `20261102_181322` with 53 completed cycles.

Run from the repository root:

```bash
cd live
python live_demo.py
```

The prototype sends real keyboard and mouse inputs. OpenArena should be running locally before starting the script. The final report treated successful pausing as an attempted procedure rather than an independently verified condition.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

Required Python packages are listed in `requirements.txt`. Ollama and OpenArena are external dependencies and are not installed by pip.

## Final evaluation configuration

Live prototype configuration used in the retained evaluation:

- model: `qwen3-vl:2b-instruct`
- local inference runtime: Ollama
- hardware used in the study: Apple M1, 8 GB memory
- maximum model-input image dimension: 768 px
- JPEG quality: 80
- context length: 4096 tokens
- temperature: 0
- request timeout: 120 s
- motion threshold: 5.0 mean absolute greyscale pixel difference
- pause-toggle delay: 0.20 s
- post-action settling delay: 0.15 s

The report records Ollama 0.34.0 from a post-experiment environment check; it should not be interpreted as definitive provenance for the exact runtime build used during every retained inference.

## Results and figures

The retained logs are included under `data/`:

- `data/controlled/final_experiment.jsonl` — 300 controlled responses
- `data/live/20261102_181322/live.jsonl` — 53 live decision cycles

Generate report figures with:

```bash
python analysis/make_capstone_figures.py \
  --controlled data/controlled/final_experiment.jsonl \
  --live data/live/20261102_181322/live.jsonl \
  --output figures \
  --fmt png \
  --no-titles
```

Generate the controlled summary table with:

```bash
python analysis/make_controlled_summary_table.py \
  --input data/controlled/final_experiment.jsonl \
  --output tables
```

Generate the live latency figure with:

```bash
python analysis/make_live_latency_figure.py \
  --input data/live/20261102_181322/live.jsonl \
  --output figures \
  --fmt png \
  --no-title
```

## Reproducibility notes

The controlled log retains the full five-state sequence, expected and evaluated actions, raw model response, explanation, parsing status, mode, scenario, repetition index and inference latency. Exact condition-specific request bodies were not stored during the run; they can be reconstructed from the retained states and the prompt-building code.

The live log retains model-reported perception fields, selected and executed actions, recovery interventions, motion differences, comparison action, reasoning text and inference latency. The original retained source-frame directory is referenced by the log but is not included in this code bundle.

## Safety / usage note

The live controller generates keyboard and mouse input through `pynput`. Review the controller code and run it only in a controlled local environment.
