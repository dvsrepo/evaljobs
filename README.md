# evaljobs

Run evals on Hugging Face GPUs. Share results and code on the Hugging Face Hub.

## Why evaljobs?

- Access hundreds of pre-built evals from [inspect_evals](https://ukgovernmentbeis.github.io/inspect_evals/)
- Write evals in Python using [Inspect AI](https://inspect.aisi.org.uk/)
- Run on HF Jobs (CPU/GPU as needed)
- Share and see live results on Spaces
- Run the same eval on different models with one command
- Discover community evals on Spaces


```mermaid
graph LR
    A[Write eval.py] --> B[evaljobs eval.py]
    B --> C[Space with results + code]
    C --> D[Share Space URL]
    D --> E[Others: evaljobs space_url]
```

## Installation

```bash
pip install git+https://github.com/dvsrepo/evaljobs.git
export HF_TOKEN=your_token_here
```

## Usage

### Run pre-built evals from inspect_evals

```bash
evaljobs inspect_evals/arc_easy \
  --model hf-inference-providers/openai/gpt-oss-120b:fastest \
  --name arc-eval
```

### Run your custom eval

```bash
evaljobs examples/midicaps_eval.py \
  --model hf/Qwen/Qwen3-0.6B \
  --name midicaps-eval \
  --flavor t4-small
```

### Run eval from a Space

```bash
evaljobs username/space-name \
  --model hf-inference-providers/openai/gpt-oss-120b:fastest \
  --name my-eval
```

### Run eval with multiple models

```bash
evaljobs inspect_evals/arc_easy \
  --model hf-inference-providers/openai/gpt-oss-20b:fastest,hf-inference-providers/openai/gpt-oss-120b:fastest \
  --name arc-eval-multi
```

## Options

- `--model`: Model to evaluate (required)
- `--name`: Base name for dataset and space (required)
- `--flavor`: Hardware flavor (default: cpu-basic)
- `--timeout`: Job timeout (default: 30m)
- `--limit`: Limit number of samples

## Model Selection

See the [Inspect AI providers documentation](https://inspect.aisi.org.uk/providers.html) for available models.

- **HF Inference Providers**: Use `--flavor cpu-basic` or omit (default)
- **Hugging Face models**: Require `--flavor` with GPU (e.g., `--flavor t4-medium`)
- **Closed models**

## How it works

1. **Custom evals**: Uploads your eval script to the Space
2. **inspect_evals**: Uses pre-installed package
3. Runs eval on HF Jobs with your chosen hardware
4. Exports results to parquet and publishes to dataset
5. Results viewable in Space and loadable as dataset

## Examples

- [inspect_evals](https://ukgovernmentbeis.github.io/inspect_evals/) - Pre-built evals for ARC, MMLU, GSM8K, HumanEval, etc.
- `examples/` - Custom eval templates
