# evaljobs

Run Inspect AI evals on Hugging Face Jobs.

## Installation

```bash
pip install -e .
```

## Usage

```bash
evaljobs examples/midicaps_eval.py \
  --model hf/frascuchon/midigen-Qwen3-0.6B \
  --space your-username/eval-logs \
  --flavor t4-small
```

## Options

- `--model`: Model to evaluate (required)
- `--space`: HF Space for logs and storage (required)
- `--flavor`: Hardware flavor (default: cpu-basic)
- `--timeout`: Job timeout (default: 30m)
- `--limit`: Limit number of samples

## How it works

1. Uploads your eval script to the Space
2. Uploads the runner script to the Space
3. Submits a job to HF Jobs that:
   - Downloads your eval script
   - Runs `inspect eval` with your model
   - Uploads bundled logs back to the Space
