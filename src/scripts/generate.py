"""Generate the dataset.

Usage:
    python generate.py \
        [--output-dir <output_dir>] \
        [--seed-tasks-path <seed_tasks_path>] \
        [--num-instructions-to-generate <num_instructions>] \
        [--model-name <model_name>] \
        [--num-prompt-instructions <num_prompt_instructions>] \
        [--request-batch-size <request_batch_size>] \
        [--num-cpus <num_cpus>]
"""

import logging

import click

from alpaka.generate_instruction import generate_instruction_following_data

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    default=".",
    show_default=True,
    help="Directory to save the generated dataset.",
)
@click.option(
    "--seed-tasks-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default="data/seed_tasks.jsonl",
    show_default=True,
    help="Path to the seed tasks file.",
)
@click.option(
    "--num-instructions-to-generate",
    type=int,
    default=100,
    show_default=True,
    help="Number of instructions to generate.",
)
@click.option(
    "--model-name",
    type=str,
    default="llama3.1:70b-text-q8_0",
    show_default=True,
    help="The Ollama model ID of the model to use for generation. Must be a base "
    "model, not a finetuned one.",
)
@click.option(
    "--num-prompt-instructions",
    type=int,
    default=3,
    show_default=True,
    help="Number of instructions to use as prompts for each generated instruction.",
)
@click.option(
    "--request-batch-size",
    type=int,
    default=5,
    show_default=True,
    help="Number of requests to send to the model at once.",
)
@click.option(
    "--num-cpus",
    type=int,
    default=1,
    show_default=True,
    help="Number of CPUs to use for parallel processing.",
)
def generate(
    output_dir: str,
    seed_tasks_path: str,
    num_instructions_to_generate: int,
    model_name: str,
    num_prompt_instructions: int,
    request_batch_size: int,
    num_cpus: int,
) -> None:
    """Generate the dataset.

    Args:
        output_dir:
            Directory to save the generated dataset.
        seed_tasks_path:
            Path to the seed tasks file.
        num_instructions_to_generate:
            Number of instructions to generate.
        model_name:
            The Ollama model ID of the model to use for generation. Must be a base
            model, not a finetuned one.
        num_prompt_instructions:
            Number of instructions to use as prompts for each generated instruction.
        request_batch_size:
            Number of requests to send to the model at once.
        num_cpus:
            Number of CPUs to use for parallel processing.
    """
    generate_instruction_following_data(
        output_dir=output_dir,
        seed_tasks_path=seed_tasks_path,
        num_instructions_to_generate=num_instructions_to_generate,
        model_name=model_name,
        num_prompt_instructions=num_prompt_instructions,
        request_batch_size=request_batch_size,
        num_cpus=num_cpus,
    )


if __name__ == "__main__":
    generate()
