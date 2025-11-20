#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "inspect-ai @ git+https://github.com/dvsrepo/inspect_ai.git@fallback-to-modified-for-hf-fs",
#     "datasets",
#     "openai",
#     "transformers",
#     "accelerate",
#     "huggingface_hub",
#     "inspect-evals",
#     "pandas",
#     "pyarrow",
# ]
# ///

import os
import sys
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from inspect_ai.analysis import evals_df, samples_df


def export_logs_to_parquet(log_dir: str, dataset_repo: str) -> None:
    from huggingface_hub import HfApi

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    api = HfApi(token=hf_token)

    repo_id = (
        dataset_repo.replace("datasets/", "")
        if dataset_repo.startswith("datasets/")
        else dataset_repo
    )

    evals = evals_df(logs=log_dir)
    samples = samples_df(logs=log_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        evals_path = Path(tmpdir) / "evals.parquet"
        samples_path = Path(tmpdir) / "samples.parquet"

        evals.to_parquet(evals_path, index=False, engine="pyarrow")
        samples.to_parquet(samples_path, index=False, engine="pyarrow")

        api.upload_file(
            path_or_fileobj=str(evals_path),
            path_in_repo="evals.parquet",
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
        )

        api.upload_file(
            path_or_fileobj=str(samples_path),
            path_in_repo="samples.parquet",
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
        )


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: eval_runner.py <eval_ref> <model> <dataset_repo> [--inspect-evals] [extra_args...]")
        sys.exit(1)

    eval_ref = sys.argv[1]
    model = sys.argv[2]
    dataset_repo = sys.argv[3]

    is_inspect_evals = "--inspect-evals" in sys.argv
    extra_args = [arg for arg in sys.argv[4:] if arg != "--inspect-evals"]

    if not dataset_repo.startswith("datasets/"):
        dataset_repo = f"datasets/{dataset_repo}"
    log_dir = f"hf://{dataset_repo}/logs"

    is_eval_set = "," in model

    if is_inspect_evals:
        eval_target = eval_ref
        cleanup_file = None
    else:
        print("Downloading eval script...")
        with urllib.request.urlopen(eval_ref) as response:
            eval_code = response.read().decode("utf-8")

        eval_filename = "downloaded_eval.py"
        with open(eval_filename, "w") as f:
            f.write(eval_code)

        eval_target = eval_filename
        cleanup_file = eval_filename

    try:
        if is_eval_set:
            print("Running evaluation set...")
            cmd = [
                "inspect",
                "eval-set",
                eval_target,
                "--model",
                model,
                "--log-dir",
                log_dir,
                "--log-shared",
                "--log-buffer",
                "100",
            ]
        else:
            print("Running evaluation...")
            cmd = [
                "inspect",
                "eval",
                eval_target,
                "--model",
                model,
                "--log-dir",
                log_dir,
                "--log-shared",
                "--log-buffer",
                "100",
            ]
        cmd.extend(extra_args)

        subprocess.run(cmd, check=True)

        print("Exporting logs to parquet...")
        try:
            export_logs_to_parquet(log_dir, dataset_repo)
        except Exception as e:
            print(f"Warning: Could not export to parquet: {e}")

    finally:
        if cleanup_file and os.path.exists(cleanup_file):
            os.unlink(cleanup_file)
