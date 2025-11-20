#!/usr/bin/env python3

import argparse
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from huggingface_hub import run_uv_job, HfApi

from .dataset import (
    ensure_dataset_exists,
    create_dataset_readme,
)
from .docker_space import create_docker_space


def generate_readme(args, extra_args, eval_ref, is_inspect_evals, is_from_space, space_id):
    cmd_lines = [f"evaljobs {args.script}"]
    cmd_lines.append(f"  --model {args.model}")
    cmd_lines.append(f"  --name {args.name}")
    if args.flavor != "cpu-basic":
        cmd_lines.append(f"  --flavor {args.flavor}")
    if args.timeout != "30m":
        cmd_lines.append(f"  --timeout {args.timeout}")
    if args.limit:
        cmd_lines.append(f"  --limit {args.limit}")

    if extra_args:
        i = 0
        while i < len(extra_args):
            arg = extra_args[i]
            if (
                arg.startswith("--")
                and i + 1 < len(extra_args)
                and not extra_args[i + 1].startswith("--")
            ):
                cmd_lines.append(f"  {arg} {extra_args[i + 1]}")
                i += 2
            else:
                cmd_lines.append(f"  {arg}")
                i += 1

    evaljobs_cmd = " \\\n".join(cmd_lines)

    eval_target = eval_ref if is_inspect_evals else "eval.py"
    inspect_cmd_lines = [f"inspect eval {eval_target}"]
    inspect_cmd_lines.append(f"  --model {args.model}")
    if args.limit:
        inspect_cmd_lines.append(f"  --limit {args.limit}")
    inspect_cmd_lines.append("  --log-shared")
    inspect_cmd_lines.append("  --log-buffer 100")

    if extra_args:
        i = 0
        while i < len(extra_args):
            arg = extra_args[i]
            if (
                arg.startswith("--")
                and i + 1 < len(extra_args)
                and not extra_args[i + 1].startswith("--")
            ):
                inspect_cmd_lines.append(f"  {arg} {extra_args[i + 1]}")
                i += 2
            else:
                inspect_cmd_lines.append(f"  {arg}")
                i += 1

    inspect_cmd = " \\\n".join(inspect_cmd_lines)

    if is_inspect_evals:
        eval_name = eval_ref.replace("inspect_evals/", "")
        script_ref = eval_ref
    elif is_from_space:
        eval_name = "eval.py"
        script_ref = args.script if "/" in args.script and not args.script.startswith("http") else args.script.replace("https://huggingface.co/spaces/", "")
    else:
        eval_name = Path(args.script).stem
        script_ref = f"https://huggingface.co/spaces/{space_id}"

    title = eval_name.replace("_", " ").replace("-", " ").title()

    readme_content = f"""---
title: {title}
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
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
  --name <your-name> \\
  --flavor {args.flavor}
```

## Inspect eval command

The eval was executed with:

```bash
{inspect_cmd}
```
"""

    return readme_content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script")
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--flavor",
        default="cpu-basic",
        choices=[
            "cpu-basic",
            "cpu-upgrade",
            "cpu-xl",
            "t4-small",
            "t4-medium",
            "l4x1",
            "l4x4",
            "a10g-small",
            "a10g-large",
            "a10g-largex2",
            "a10g-largex4",
            "a100-large",
            "h100",
            "h100x8",
        ],
    )
    parser.add_argument("--timeout", default="30m")

    args, extra_args = parser.parse_known_args()

    is_inspect_evals = args.script.startswith("inspect_evals/")
    is_local_file = Path(args.script).exists()
    is_http_url = args.script.startswith("http")
    is_space_ref = "/" in args.script and not is_http_url and not is_inspect_evals and not is_local_file

    if (
        not is_inspect_evals
        and not is_http_url
        and not is_space_ref
        and not is_local_file
    ):
        print(f"Error: Eval script '{args.script}' not found", file=sys.stderr)
        sys.exit(1)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable not set", file=sys.stderr)
        sys.exit(1)

    try:
        api = HfApi(token=hf_token)
        user_info = api.whoami()
        username = user_info["name"]

        dataset_repo = f"datasets/{username}/{args.name}"
        space_id = f"{username}/{args.name}"

        print("Setting up dataset...")
        dataset_repo = ensure_dataset_exists(
            dataset_repo=dataset_repo,
            hf_token=hf_token,
        )

        create_dataset_readme(
            dataset_repo=dataset_repo,
            hf_token=hf_token,
            name=args.name,
            model=args.model,
            space_id=space_id,
            script=args.script,
            is_inspect_evals=is_inspect_evals,
        )

        print("Creating space...")
        create_docker_space(
            space_id=space_id,
            dataset_repo=dataset_repo,
            hf_token=hf_token,
        )

        print("Uploading files...")

        if is_inspect_evals:
            eval_ref = args.script
        elif is_http_url or is_space_ref:
            if is_http_url:
                source_space_id = args.script.replace(
                    "https://huggingface.co/spaces/", ""
                ).rstrip("/")
            else:
                source_space_id = args.script

            try:
                files = api.list_repo_files(repo_id=source_space_id, repo_type="space")
            except Exception as e:
                print(
                    f"✗ Error: Could not access Space {source_space_id}: {e}",
                    file=sys.stderr,
                )
                sys.exit(1)

            eval_files = [
                f for f in files if f.startswith("eval") and f.endswith(".py")
            ]

            if not eval_files:
                print(
                    f"✗ Error: No eval script found in Space {source_space_id}",
                    file=sys.stderr,
                )
                sys.exit(1)

            source_eval_url = f"https://huggingface.co/spaces/{source_space_id}/resolve/main/{eval_files[0]}"

            with urllib.request.urlopen(source_eval_url) as response:
                eval_content = response.read()

            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".py", delete=False
            ) as tmp:
                tmp.write(eval_content)
                tmp_path = tmp.name

            api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo="eval.py",
                repo_id=space_id,
                repo_type="space",
            )
            os.unlink(tmp_path)
            eval_ref = f"https://huggingface.co/spaces/{space_id}/resolve/main/eval.py"
        else:
            # For local files, upload as eval.py to destination Space
            api.upload_file(
                path_or_fileobj=args.script,
                path_in_repo="eval.py",
                repo_id=space_id,
                repo_type="space",
            )
            eval_ref = f"https://huggingface.co/spaces/{space_id}/resolve/main/eval.py"

        runner_path = Path(__file__).parent / "runner.py"
        api.upload_file(
            path_or_fileobj=str(runner_path),
            path_in_repo="runner.py",
            repo_id=space_id,
            repo_type="space",
        )
        runner_url = f"https://huggingface.co/spaces/{space_id}/resolve/main/runner.py"

        readme_content = generate_readme(
            args, extra_args, eval_ref, is_inspect_evals, is_http_url or is_space_ref, space_id
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(readme_content)
            tmp_path = tmp.name

        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=space_id,
            repo_type="space",
        )
        os.unlink(tmp_path)

        print("Submitting job...")
        script_args = [eval_ref, args.model, dataset_repo]
        if is_inspect_evals:
            script_args.append("--inspect-evals")

        if args.limit:
            script_args.extend(["--limit", str(args.limit)])

        if extra_args:
            script_args.extend(extra_args)

        job_info = run_uv_job(
            script=runner_url,
            script_args=script_args,
            flavor=args.flavor,
            timeout=args.timeout,
            secrets={"HF_TOKEN": hf_token},
        )

        repo_id = dataset_repo.replace("datasets/", "")
        print(f"Monitor eval job: {job_info.url}")
        print(f"Browse live eval results: https://huggingface.co/spaces/{space_id}")
        print(f"View eval dataset: https://huggingface.co/datasets/{repo_id}")
        

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
