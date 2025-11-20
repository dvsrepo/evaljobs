import io
from pathlib import Path
from huggingface_hub import HfApi


def ensure_dataset_exists(
    dataset_repo: str,
    hf_token: str,
) -> str:
    api = HfApi(token=hf_token)

    repo_id = (
        dataset_repo.replace("datasets/", "")
        if dataset_repo.startswith("datasets/")
        else dataset_repo
    )

    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
    except Exception:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=False,
        )

    try:
        api.upload_file(
            path_or_fileobj=io.BytesIO(b"# This file ensures the logs directory exists\n"),
            path_in_repo="logs/.gitkeep",
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
        )
    except Exception:
        pass

    return dataset_repo


def get_log_dir_for_dataset(dataset_repo: str) -> str:
    if not dataset_repo.startswith("datasets/"):
        dataset_repo = f"datasets/{dataset_repo}"

    return f"hf://{dataset_repo}/logs"


def create_dataset_readme(
    dataset_repo: str,
    hf_token: str,
    name: str,
    model: str,
    space_id: str,
    script: str,
    is_inspect_evals: bool,
    evaljobs_cmd: str,
    inspect_cmd: str,
    script_ref: str,
    flavor: str,
) -> None:
    api = HfApi(token=hf_token)

    repo_id = (
        dataset_repo.replace("datasets/", "")
        if dataset_repo.startswith("datasets/")
        else dataset_repo
    )

    if is_inspect_evals:
        script_text = f"the eval `{script}` from [Inspect Evals](https://ukgovernmentbeis.github.io/inspect_evals/)"
    else:
        script_name = Path(script).name
        script_text = f"the eval script [{script_name}](https://huggingface.co/spaces/{space_id}/blob/main/eval.py)"

    readme_content = f"""---
configs:
  - config_name: default
    data_files:
      - split: evals
        path: evals.parquet
      - split: samples
        path: samples.parquet
---

# {name} Evaluation Results

Eval created with [evaljobs](https://github.com/dvsrepo/evaljobs).

This dataset contains evaluation results for the model(s) `{model}` using {script_text}.

To browse the results interactively, visit [this Space](https://huggingface.co/spaces/{space_id}).

## Command

This eval was run with:

```bash
{evaljobs_cmd}
```

## Run with other models

To run this eval with a different model, use:

```bash
pip install git+https://github.com/dvsrepo/evaljobs.git
export HF_TOKEN=your_token_here

evaljobs {script_ref} \\
  --model <your-model> \\
  --name <your-name> \\
  --flavor {flavor}
```

**Note:** For model selection, see the [Inspect AI providers documentation](https://inspect.aisi.org.uk/providers.html). Common examples:
- Hugging Face models: `hf/meta-llama/Llama-3.1-8B-Instruct` (requires `--flavor` with GPU, e.g., `--flavor t4-medium`)
- HF Inference Providers: `hf-inference-providers/openai/gpt-oss-120b:fastest` (use `--flavor cpu-basic` or omit)

## Inspect eval command

The eval was executed with:

```bash
{inspect_cmd}
```

## Splits

- **evals**: Evaluation runs metadata (one row per evaluation run)
- **samples**: Sample-level data (one row per sample)

## Loading

```python
from datasets import load_dataset

evals = load_dataset('{repo_id}', split='evals')
samples = load_dataset('{repo_id}', split='samples')
```
"""

    api.upload_file(
        path_or_fileobj=io.BytesIO(readme_content.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token,
    )
