# evaljobs

Evaluate open models on Hugging Face GPUs.

Write your own evals using [Inspect](https://inspect.aisi.org.uk/) or use [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals), which gives you instant access to hundreds of pre-built evaluations, including ARC, MMLU, GSM8K, HumanEval, GPQA, and more.

## Installation

```bash
pip install git+https://github.com/dvsrepo/evaljobs.git
```

## Usage

### Run inspect_evals (hundreds of benchmarks available)
```bash
# ARC Easy
evaljobs inspect_evals/arc_easy \
  --model hf/Qwen/Qwen3-0.6B \
  --space your-username/arc_easy-Qwen3-0.6B \
  --flavor t4-small

# MMLU, GSM8K, HumanEval, GPQA, etc.
evaljobs inspect_evals/mmlu \
  --model hf/Qwen/Qwen3-0.6B \
  --space your-username/mmlu-Qwen3-0.6B \  
  --flavor t4-small
```

### Run custom eval scripts
```bash
evaljobs examples/midicaps_eval.py \
  --model hf/Qwen/Qwen3-0.6B \
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

1. For custom eval scripts: uploads your eval script to the Space. For inspect_evals: uses the inspect_evals path directly
2. Submits a job to HF Jobs that runs the evaluation
3. Results are uploaded to your Space where you can browse, share, and analyze them

## TODO

- [ ] Support `eval-set`
- [ ] Test/support vLLM
- [ ] Test/support vlm's
