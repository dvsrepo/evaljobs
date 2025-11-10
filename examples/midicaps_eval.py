"""Inspect eval for MIDI caps benchmark."""
from typing import Any
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import ChatMessageUser, ContentText
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate


@task
def midicaps_eval():
    return Task(
        dataset=hf_dataset(
            path="dvilasuero/midicaps_benchmark",
            split="small_test",
            sample_fields=record_to_sample,
            shuffle=True,
        ),
        solver=generate(),
        scorer=model_graded_fact(
            partial_credit=True,
            model="hf-inference-providers/Qwen/Qwen3-32B:fastest"
        )
    )


def record_to_sample(record: dict[str, Any]) -> Sample:
    message = [
        ChatMessageUser(
            content=[
                ContentText(text=record['caption']),
            ]
        )
    ]
    return Sample(
        input=message,
        target=record["condensed_sequence"]
    )
