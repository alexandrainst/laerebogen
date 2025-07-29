"""Correcting spelling mistakes in text."""

import logging
from copy import deepcopy

from tqdm.auto import tqdm

from laerebogen.filtering import keep_instruction

from .data_models import InstructionSample
from .vllm_utils import generate_text_with_vllm, load_vllm_model

logger = logging.getLogger(__name__)


CORRECTION_PROMPT_TEMPLATE = (
    "Skriv følgende #Original Tekst# om, hvor du retter all stavefejl og "
    "grammatiske fejl. Du må kun skrive på dansk, du må ikke ændre betydningen af "
    "instruktionen, og du skal kun skrive teksten uden yderligere tekst. Du skal "
    "skrive den rettede tekst under #Rettet Tekst#. Hvis teksten er '<empty>', så "
    "skal du også skrive '<empty>' under #Rettet Tekst#.\n\n"
    "#Original Tekst#:\n"
    "{text}\n\n"
    "#Rettet Tekst#:\n"
)


def correct_instructions(
    instructions: list[InstructionSample], model_id: str
) -> list[InstructionSample]:
    """Correct spelling mistakes using an instruction-tuned large language model.

    Args:
        instructions:
            The instructions to correct.
        model_id:
            The model ID of the instruction-tuned large language model to use for
            correction.

    Returns:
        The corrected instructions.
    """
    # Load the model and tokenizer
    logger.info(f"Loading model {model_id!r} for correcting instructions...")
    model = load_vllm_model(model_id=model_id)
    tokenizer = model.get_tokenizer()

    # Copy the instructions to avoid modifying the original ones
    corrected_instructions = deepcopy(instructions)

    # Correct the instructions
    logger.info("Correcting instructions...")
    prompts = [
        CORRECTION_PROMPT_TEMPLATE.format(
            text=instruction.instruction if instruction.instruction else "<empty>"
        )
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
    breakpoint()
    responses = generate_text_with_vllm(prompts=prompts, model=model)
    for instruction, response in zip(corrected_instructions, responses):
        if response.done_reason == "stop":
            instruction.instruction = (
                response.completion.strip()
                if response.completion.strip() != "<empty>"
                else ""
            )

    # Correct the inputs
    logger.info("Correcting inputs...")
    prompts = [
        CORRECTION_PROMPT_TEMPLATE.format(
            text=instruction.input if instruction.input else "<empty>"
        )
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
    responses = generate_text_with_vllm(prompts=prompts, model=model)
    for instruction, response in zip(corrected_instructions, responses):
        if response.done_reason == "stop":
            instruction.input = (
                response.completion.strip()
                if response.completion.strip() != "<empty>"
                else ""
            )

    # Correct the outputs
    logger.info("Correcting outputs...")
    prompts = [
        CORRECTION_PROMPT_TEMPLATE.format(
            text=instruction.output if instruction.output else "<empty>"
        )
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
    responses = generate_text_with_vllm(prompts=prompts, model=model)
    for instruction, response in zip(corrected_instructions, responses):
        if response.done_reason == "stop":
            instruction.output = (
                response.completion.strip()
                if response.completion.strip() != "<empty>"
                else ""
            )

    # Filter the corrected instructions
    logger.info("Filtering corrected instructions...")
    corrected_instructions = [
        instruction
        for instruction in tqdm(
            iterable=corrected_instructions,
            desc="Filtering corrected instructions",
            unit="instruction",
        )
        if keep_instruction(instruction_sample=instruction)
    ]

    return corrected_instructions
