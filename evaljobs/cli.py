#!/usr/bin/env python3
"""CLI to run Inspect evals on Hugging Face Jobs."""
import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import run_uv_job, HfApi


def main():
    parser = argparse.ArgumentParser(
        description="Run Inspect evals on Hugging Face Jobs"
    )
    parser.add_argument("script", help="Path to eval script")
    parser.add_argument("--model", required=True, help="Model to evaluate")
    parser.add_argument("--space", required=True, help="HF Space ID for logs and storage")
    parser.add_argument("--log-dir", default="./logs", help="Log directory (default: ./logs)")
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument(
        "--flavor",
        default="cpu-basic",
        choices=[
            "cpu-basic", "cpu-upgrade", "cpu-xl",
            "t4-small", "t4-medium",
            "l4x1", "l4x4",
            "a10g-small", "a10g-large", "a10g-largex2", "a10g-largex4",
            "a100-large",
            "h100", "h100x8"
        ],
        help="Hardware flavor (default: cpu-basic)"
    )
    parser.add_argument("--timeout", default="30m", help="Job timeout (default: 30m)")

    args = parser.parse_args()

    if not args.script.startswith("http") and not Path(args.script).exists():
        print(f"✗ Error: Eval script '{args.script}' not found", file=sys.stderr)
        sys.exit(1)

    print("Running eval on HF Jobs...")
    print(f"  Script: {args.script}")
    print(f"  Model: {args.model}")
    print(f"  Space: {args.space}")
    print(f"  Flavor: {args.flavor}")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("✗ Error: HF_TOKEN environment variable not set", file=sys.stderr)
        print("Set it with: export HF_TOKEN=your_token", file=sys.stderr)
        sys.exit(1)

    try:
        api = HfApi(token=hf_token)

        print("\n[1/3] Setting up Space...")
        api.create_repo(
            repo_id=args.space,
            repo_type="space",
            exist_ok=True,
            space_sdk="static",
        )

        print("[2/3] Uploading files...")
        eval_filename = Path(args.script).name
        api.upload_file(
            path_or_fileobj=args.script,
            path_in_repo=f"eval_{eval_filename}",
            repo_id=args.space,
            repo_type="space",
        )
        eval_url = f"https://huggingface.co/spaces/{args.space}/resolve/main/eval_{eval_filename}"

        runner_path = Path(__file__).parent / "runner.py"
        api.upload_file(
            path_or_fileobj=str(runner_path),
            path_in_repo="runner.py",
            repo_id=args.space,
            repo_type="space",
        )
        runner_url = f"https://huggingface.co/spaces/{args.space}/resolve/main/runner.py"

        print("[3/3] Submitting job...")
        script_args = [eval_url, args.model, args.space, args.log_dir]

        if args.limit:
            script_args.extend(["--limit", str(args.limit)])

        job_info = run_uv_job(
            script=runner_url,
            script_args=script_args,
            flavor=args.flavor,
            timeout=args.timeout,
            secrets={"HF_TOKEN": hf_token}
        )

        print("\n✓ Job submitted successfully!")
        print(f"  Job ID: {job_info.id}")
        print(f"  Status: {job_info.status}")
        print(f"\nMonitor: {job_info.url}")

    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
