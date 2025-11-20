import os
import tempfile
from typing import Optional
from huggingface_hub import HfApi


README_TEMPLATE = """---
title: {title}
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "latest"
pinned: false
---

# {title}

Live log viewer for eval results stored in [{dataset_repo}](https://huggingface.co/{dataset_repo}).

This Space runs `inspect view` to display real-time evaluation logs from the dataset.

## View Logs

Logs are automatically displayed from: `{log_dir}`

## Dataset

Results are stored in: [{dataset_repo}](https://huggingface.co/{dataset_repo})
"""


def create_docker_space(
    space_id: str,
    dataset_repo: str,
    hf_token: str,
    title: Optional[str] = None,
    template_space: str = "dvilasuero/evaljobs_docker_template",
) -> str:
    api = HfApi(token=hf_token)

    log_dir = f"hf://{dataset_repo}/logs"

    if not title:
        title = space_id.split("/")[-1].replace("-", " ").replace("_", " ").title()

    try:
        api.repo_info(repo_id=space_id, repo_type="space")
        space_exists = True
    except Exception:
        space_exists = False

    if not space_exists:
        api.duplicate_space(
            from_id=template_space,
            to_id=space_id,
            private=False,
            token=hf_token,
        )

    try:
        api.add_space_variable(
            repo_id=space_id,
            key="LOG_DIR",
            value=log_dir,
        )
    except Exception:
        api.delete_space_variable(repo_id=space_id, key="LOG_DIR")
        api.add_space_variable(
            repo_id=space_id,
            key="LOG_DIR",
            value=log_dir,
        )

    readme_content = README_TEMPLATE.format(
        title=title,
        dataset_repo=dataset_repo.replace("datasets/", ""),
        log_dir=log_dir,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(readme_content)
        readme_path = tmp.name

    try:
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=space_id,
            repo_type="space",
        )
    finally:
        if os.path.exists(readme_path):
            os.unlink(readme_path)

    return space_id
