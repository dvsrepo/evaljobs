from inspect_ai import task
from inspect_evals.simpleqa.simpleqa import simpleqa_verified


@task
def simpleqa_verified_custom():
    return simpleqa_verified(
        grader_model="hf-inference-providers/openai/gpt-oss-120b:fastest",
    )