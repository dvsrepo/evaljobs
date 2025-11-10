#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "inspect-ai",
#     "datasets",
#     "openai",
#     "transformers",
#     "accelerate",
#     "huggingface_hub",
# ]
# ///
"""Runner that downloads an eval script and executes it using inspect CLI."""
import os
import sys
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from huggingface_hub import HfApi
from inspect_ai.log import bundle_log_dir


def bundle_and_upload_to_space(log_dir: str, hf_space_id: str, hf_token: str):
    """Bundle logs and upload to HF Space."""
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"Warning: Log directory '{log_dir}' does not exist")
        return

    with tempfile.TemporaryDirectory() as temp_bundle_dir:
        bundle_output_dir = os.path.join(temp_bundle_dir, "bundle")

        print(f"Bundling logs from {log_dir}...")
        bundle_log_dir(log_dir=log_dir, output_dir=bundle_output_dir, overwrite=True)

        api = HfApi(token=hf_token)

        print(f"Uploading to Space {hf_space_id}...")
        uploaded_count = 0
        for root, _, files in os.walk(bundle_output_dir):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, bundle_output_dir)
                path_in_repo = rel_path.replace(os.sep, "/")

                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=path_in_repo,
                    repo_id=hf_space_id,
                    repo_type="space",
                )
                uploaded_count += 1

        print(f"Uploaded {uploaded_count} files")
        print(f"View at: https://huggingface.co/spaces/{hf_space_id}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: eval_runner.py <eval_url> <model> <space_id> [log_dir] [extra_args...]")
        sys.exit(1)

    eval_script_url = sys.argv[1]
    model = sys.argv[2]
    hf_space_id = sys.argv[3]
    log_dir = sys.argv[4] if len(sys.argv) > 4 else "./logs"
    extra_args = sys.argv[5:] if len(sys.argv) > 5 else []

    print(f"Downloading eval from {eval_script_url}...")
    with urllib.request.urlopen(eval_script_url) as response:
        eval_code = response.read().decode('utf-8')

    eval_filename = "downloaded_eval.py"
    with open(eval_filename, 'w') as f:
        f.write(eval_code)

    try:
        print(f"Running inspect eval with model {model}...")
        cmd = [
            "inspect", "eval", eval_filename,
            "--model", model,
            "--log-dir", log_dir,
        ]
        cmd.extend(extra_args)

        print(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        print(f"\nUploading logs to {hf_space_id}...")
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("Warning: HF_TOKEN not set, skipping upload")
        else:
            bundle_and_upload_to_space(log_dir, hf_space_id, hf_token)

    finally:
        if os.path.exists(eval_filename):
            os.unlink(eval_filename)
