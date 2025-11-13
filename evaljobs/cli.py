#!/usr/bin/env python3
"""CLI to run Inspect evals on Hugging Face Jobs."""
import argparse
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from huggingface_hub import run_uv_job, HfApi


def generate_readme(args, extra_args, eval_ref, is_inspect_evals):
    """Generate README content for the eval space."""
    # Build the evaljobs command - group argument-value pairs
    cmd_lines = [f"evaljobs {args.script}"]
    cmd_lines.append(f"  --model {args.model}")
    cmd_lines.append(f"  --space {args.space}")
    if args.flavor != "cpu-basic":
        cmd_lines.append(f"  --flavor {args.flavor}")
    if args.timeout != "30m":
        cmd_lines.append(f"  --timeout {args.timeout}")
    if args.limit:
        cmd_lines.append(f"  --limit {args.limit}")
    
    # Handle extra args - they might be flags or key-value pairs
    if extra_args:
        i = 0
        while i < len(extra_args):
            arg = extra_args[i]
            if arg.startswith("--") and i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
                # It's a flag with a value
                cmd_lines.append(f"  {arg} {extra_args[i + 1]}")
                i += 2
            else:
                # It's a standalone flag
                cmd_lines.append(f"  {arg}")
                i += 1
    
    evaljobs_cmd = " \\\n".join(cmd_lines)
    
    # Build the inspect eval command (what will be run) - group argument-value pairs
    eval_target = eval_ref if is_inspect_evals else "eval.py"
    inspect_cmd_lines = [f"inspect eval {eval_target}"]
    inspect_cmd_lines.append(f"  --model {args.model}")
    inspect_cmd_lines.append(f"  --log-dir {args.log_dir}")
    if args.limit:
        inspect_cmd_lines.append(f"  --limit {args.limit}")
    
    # Handle extra args - they might be flags or key-value pairs
    if extra_args:
        i = 0
        while i < len(extra_args):
            arg = extra_args[i]
            if arg.startswith("--") and i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
                # It's a flag with a value
                inspect_cmd_lines.append(f"  {arg} {extra_args[i + 1]}")
                i += 2
            else:
                # It's a standalone flag
                inspect_cmd_lines.append(f"  {arg}")
                i += 1
    
    inspect_cmd = " \\\n".join(inspect_cmd_lines)
    
    # Determine eval name and script reference for "Run with other models"
    if is_inspect_evals:
        eval_name = eval_ref.replace("inspect_evals/", "")
        script_ref = eval_ref
    elif args.script.startswith("http"):
        eval_name = "eval.py"
        # Use current space URL since eval.py was uploaded here
        script_ref = f"https://huggingface.co/spaces/{args.space}"
    else:
        eval_name = Path(args.script).stem
        # Use current space URL since eval.py was uploaded here
        script_ref = f"https://huggingface.co/spaces/{args.space}"
    
    # Format title nicely (replace underscores with spaces, title case)
    title = eval_name.replace("_", " ").replace("-", " ").title()
    
    readme_content = f"""---
title: {title}
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: static
sdk_version: "latest"
pinned: false
---

# {eval_name}

This eval was run using [evaljobs](https://github.com/dvsrepo/evaljobs).

## Command

```bash
{evaljobs_cmd}
```

## Run with other models

To run this eval with a different model, use:

```bash
evaljobs {script_ref} \\
  --model <your-model> \\
  --space <your-space> \\
  --flavor {args.flavor}
```

Replace `<your-model>` with your model identifier (e.g., `hf/Qwen/Qwen3-1.7B`) and `<your-space>` with your desired Space ID.

## Inspect eval command

The eval was executed with:

```bash
{inspect_cmd}
```
"""
    
    return readme_content


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

    args, extra_args = parser.parse_known_args()

    # Check if it's an inspect_evals path or a local file
    is_inspect_evals = args.script.startswith("inspect_evals/")

    if not is_inspect_evals and not args.script.startswith("http") and not Path(args.script).exists():
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

        if is_inspect_evals:
            # For inspect_evals, pass the path directly (no upload needed)
            eval_ref = args.script
        elif args.script.startswith("http"):
            # For Space URLs, download and re-upload as eval.py
            source_space_id = args.script.replace("https://huggingface.co/spaces/", "").rstrip("/")

            files = api.list_repo_files(repo_id=source_space_id, repo_type="space")
            eval_files = [f for f in files if f.startswith("eval") and f.endswith(".py")]

            if not eval_files:
                print(f"✗ Error: No eval script found in Space {source_space_id}", file=sys.stderr)
                sys.exit(1)

            source_eval_url = f"https://huggingface.co/spaces/{source_space_id}/resolve/main/{eval_files[0]}"

            with urllib.request.urlopen(source_eval_url) as response:
                eval_content = response.read()

            with tempfile.NamedTemporaryFile(mode='wb', suffix='.py', delete=False) as tmp:
                tmp.write(eval_content)
                tmp_path = tmp.name

            api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo="eval.py",
                repo_id=args.space,
                repo_type="space",
            )
            os.unlink(tmp_path)
            eval_ref = f"https://huggingface.co/spaces/{args.space}/resolve/main/eval.py"
        else:
            # For local files, upload as eval.py to destination Space
            api.upload_file(
                path_or_fileobj=args.script,
                path_in_repo="eval.py",
                repo_id=args.space,
                repo_type="space",
            )
            eval_ref = f"https://huggingface.co/spaces/{args.space}/resolve/main/eval.py"

        runner_path = Path(__file__).parent / "runner.py"
        api.upload_file(
            path_or_fileobj=str(runner_path),
            path_in_repo="runner.py",
            repo_id=args.space,
            repo_type="space",
        )
        runner_url = f"https://huggingface.co/spaces/{args.space}/resolve/main/runner.py"

        # Generate and upload README
        readme_content = generate_readme(args, extra_args, eval_ref, is_inspect_evals)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp:
            tmp.write(readme_content)
            tmp_path = tmp.name
        
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=args.space,
            repo_type="space",
        )
        os.unlink(tmp_path)

        print("[3/3] Submitting job...")
        script_args = [eval_ref, args.model, args.space, args.log_dir]
        if is_inspect_evals:
            script_args.append("--inspect-evals")

        if args.limit:
            script_args.extend(["--limit", str(args.limit)])

        # Pass through any extra args to inspect
        if extra_args:
            script_args.extend(extra_args)

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
