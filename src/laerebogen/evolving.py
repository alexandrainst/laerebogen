"""Evolving an existing instruction-following dataset.

This is the method described in [1], using an existing instruction-tuned language model
to generate new instructions based on an existing dataset, to make the dataset more
complicated and diverse.

[1] https://doi.org/10.48550/arXiv.2304.12244
"""

import logging
import random
import typing as t
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer
from tqdm.auto import tqdm

from .data_models import InstructionSample
from .filtering import keep_instruction
from .vllm_utils import generate_text_with_vllm

if t.TYPE_CHECKING:
    from vllm import LLM

logger = logging.getLogger(__name__)


METHODS = dict(
    add_constraints="Tilføj venligst endnu en begrænsning/et krav til "
    "#Original Prompt#.",
    deepen="Hvis #Original Prompt# indeholder forespørgsler om bestemte emner, kan "
    "dybden og bredden af forespørgslen øges.",
    concretise="Erstat venligst generelle begreber med mere specifikke begreber.",
    increase_reasoning_steps="Hvis #Original Prompt# kan løses med blot et par enkle "
    "tankeprocesser, kan du omskrive den, så den eksplicit beder om at ræsonnere i "
    "flere trin.",
    complicate_input="Du skal tilføje {format} data som inputdata til "
    "#Omskrevet Prompt#.",
)


FORMATS = [
    "XML",
    "HTML",
    "JSON",
    "CSV",
    "SQL",
    "Markdown",
    "Python",
    "JavaScript",
    "Java",
    "C++",
    "C#",
    "Golang",
    "Rust",
    "PHP",
    "Ruby",
    "Shell",
    "Bash",
    "PowerShell",
    "TypeScript",
    "Kotlin",
    "Swift",
    "Rlang",
    "MATLAB",
    "LaTeX",
    "YAML",
    "TOML",
]


def evolve_instructions(
    instructions: list[InstructionSample],
    model: "LLM",
    rewriter_prompt_path: str,
    creator_prompt_path: str,
    num_cpus: int,
) -> list[InstructionSample]:
    """Evolve an instruction using an instruction-tuned large language model.

    Args:
        instructions:
            The instructions to evolve.
        model:
            The instruction-tuned large language model to use for evolution.
        rewriter_prompt_path:
            Path to the prompt file containing the rewriter prompt.
        creator_prompt_path:
            Path to the prompt file containing the creator prompt.
        num_cpus:
            The number of CPU cores to use for parallel processing.

    Returns:
        The evolved instructions as well as the original instructions, shuffled.
    """
    # Load the tokenizer
    tokenizer = model.get_tokenizer()

    # Load the prompts
    with Path(rewriter_prompt_path).open() as f:
        rewriter_prompt = f.read() + "\n"
    with Path(creator_prompt_path).open() as f:
        creator_prompt = f.read() + "\n"

    # Prepare the prompt templates
    templates = [
        rewriter_prompt.replace("{method}", method) for method in METHODS.values()
    ] + [creator_prompt]

    # Evolve the instructions
    logger.info("Evolving instructions...")
    prompts = [
        random.choice(templates)
        .replace("{format}", random.choice(FORMATS))
        .format(instruction=f"{instruction.instruction}\n\n{instruction.input}")
        for instruction in instructions
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [dict(role="user", content=prompt)],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in tqdm(
            iterable=prompts,
            desc="Applying chat template to prompts",
            unit="prompt",
            leave=False,
        )
    ]
    evolved_instructions = [
        InstructionSample(instruction=response.completion.strip(), input="", output="")
        for response in generate_text_with_vllm(prompts=prompts, model=model)
        if response.done_reason == "stop"
    ]

    logger.info(
        f"Computing similarity scores for {len(evolved_instructions):,} evolved "
        "instructions against the original instructions..."
    )

    # Tokenize the previous instructions, to check for similarity of the evolved
    # instructions with the previous ones
    scorer = rouge_scorer.RougeScorer(rouge_types=["rougeL"], use_stemmer=False)
    instruction_tokens = [
        scorer._tokenizer.tokenize(text=instruction.instruction)
        for instruction in tqdm(
            iterable=instructions,
            desc="Tokenising original instructions",
            unit="instruction",
            leave=False,
        )
    ]

    # Compute the similarity of the evolved instructions to all previous instructions
    previous_instructions = instructions + evolved_instructions
    for evolved_instruction in tqdm(
        iterable=evolved_instructions,
        desc="Computing similarity scores",
        unit="instruction",
        leave=False,
    ):
        # Compute the similarity scores
        new_instruction_tokens = scorer._tokenizer.tokenize(
            text=evolved_instruction.instruction
        )
        with Pool(processes=num_cpus) as p:
            rouge_scores = p.map(
                partial(rouge_scorer._score_lcs, new_instruction_tokens),
                instruction_tokens,
            )
        rouge_scores = [score.fmeasure for score in rouge_scores]

        # Update the evolved instruction with the similarity scores
        evolved_instruction.avg_similarity_score = np.mean(rouge_scores).item()
        evolved_instruction.most_similar_instructions = {
            previous_instructions[i].instruction: rouge_scores[i]
            for i in np.argsort(rouge_scores)[-10:][::-1]
        }

        # Add the tokens of the evolved instruction to the list of instruction tokens
        instruction_tokens.append(new_instruction_tokens)

    # Get the corresponding outputs
    logger.info("Generating outputs for evolved instructions...")
    prompts = [
        tokenizer.apply_chat_template(
            [dict(role="user", content=instruction.instruction)],
            add_generation_prompt=True,
            tokenize=False,
        )
        for instruction in tqdm(
            iterable=evolved_instructions,
            desc="Applying chat template to prompts",
            unit="prompt",
            leave=False,
        )
    ]
    responses = generate_text_with_vllm(prompts=prompts, model=model)
    for instruction, output in zip(evolved_instructions, responses):
        if output.done_reason == "stop":
            instruction.output = output.completion.strip()

    # Filter the evolved instructions
    logger.info("Filtering evolved instructions...")
    evolved_instructions = [
        instruction
        for instruction in tqdm(
            iterable=evolved_instructions,
            desc="Filtering evolved instructions",
            unit="instruction",
        )
        if keep_instruction(instruction_sample=instruction)
    ]

    # Shuffle the evolved samples and the original samples
    all_samples = evolved_instructions + instructions
    random.shuffle(all_samples)

    return all_samples
