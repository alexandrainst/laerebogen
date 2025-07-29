"""Evolving an existing instruction-following dataset.

This is the method described in [1], using an existing instruction-tuned language model
to generate new instructions based on an existing dataset, to make the dataset more
complicated and diverse.

[1] https://doi.org/10.48550/arXiv.2304.12244
"""

import logging
import random
from functools import partial
from multiprocessing import Pool

import numpy as np
from rouge_score import rouge_scorer
from tqdm.auto import tqdm

from .data_models import InstructionSample
from .filtering import keep_instruction
from .vllm_utils import generate_text_with_vllm, load_vllm_model

logger = logging.getLogger(__name__)


PROMPT_REWRITER_TEMPLATE = (
    "Jeg vil have dig til at fungere som prompt-omskriver.\n\n"
    "Dit mål er at omskrive en given prompt til en mere kompleks version for at "
    "gøre de berømte AI-systemer (f.eks. ChatGPT og GPT4) lidt sværere at håndtere. "
    "Men den omskrevne prompt skal være rimelig og skal kunne forstås og besvares af "
    "mennesker.\n\n"
    "Din omskrivning må ikke udelade de dele, der ikke er tekst, såsom tabellen og "
    "koden i #Original Prompt#. Og du må heller ikke udelade inputtet i "
    "#Original Prompt#.\n\n"
    "Du må kun skrive på dansk.\n\n"
    "Du SKAL komplicere den givne prompt ved hjælp af følgende metode:\n"
    "{method}\n\n"
    "Du bør gøre dit bedste for ikke at gøre den #Omskrevet Prompt# for lang, "
    "#Omskrevet Prompt# kan kun tilføje 10 til 20 ord til #givet opgave#. "
    "‘#Original Prompt#', ‘#Omskrevet Prompt#', ‘original prompt' og "
    "‘omskrevet prompt' må ikke optræde i #Omskrevet Prompt#.\n\n"
    "#Original Prompt#:\n"
    "{instruction}\n\n"
    "#Omskrevet Prompt#:\n"
)


PROMPT_CREATOR_TEMPLATE = (
    "Jeg vil have dig til at fungere som prompt-skaber.\n\n"
    "Dit mål er at lade dig inspirere af #Original Prompt# til at skabe en helt ny "
    "prompt.\n\n"
    "Denne nye prompt skal tilhøre samme domæne som #Original Prompt#, men være endnu "
    "mere sjælden.\n\n"
    "Længden og sværhedsgraden af #Skabt Prompt# skal være den samme som for den "
    "#Original Prompt#.\n\n"
    "#Skabt Prompt# skal være rimelig og skal kunne forstås og besvares af mennesker. "
    "'#Original Prompt#', '#Skabt Prompt#', 'original prompt' og 'skabt prompt' må "
    "ikke optræde i #Skabt Prompt#.\n\n"
    "Du må kun skrive på dansk.\n\n"
    "#Original Prompt#:\n"
    "{instruction}\n\n"
    "#Skabt Prompt#:\n"
)


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
    "Shell-script",
    "Bash-script",
    "PowerShell-script",
    "TypeScript",
    "Kotlin",
    "Swift",
    "R",
    "MATLAB",
    "LaTeX",
    "YAML",
    "TOML",
]


def evolve_instructions(
    instructions: list[InstructionSample], model_id: str, num_cpus: int
) -> list[InstructionSample]:
    """Evolve an instruction using an instruction-tuned large language model.

    Args:
        instructions:
            The instructions to evolve.
        model_id:
            The model ID of the instruction-tuned large language model to use for
            evolution.
        num_cpus:
            The number of CPU cores to use for parallel processing.

    Returns:
        The evolved instructions as well as the original instructions, shuffled.
    """
    # Load the model and tokenizer
    logger.info(f"Loading model {model_id!r} for evolving instructions...")
    model = load_vllm_model(model_id=model_id)
    tokenizer = model.get_tokenizer()

    # Prepare the prompt templates
    templates = [
        PROMPT_REWRITER_TEMPLATE.replace("{method}", method)
        for method in METHODS.values()
    ] + [PROMPT_CREATOR_TEMPLATE]

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
        InstructionSample(instruction=response.completion, input="", output="")
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
        for instruction in instructions
    ]

    # Compute the similarity of the evolved instructions to all previous instructions
    for evolved_instruction in evolved_instructions:
        new_instruction_tokens = scorer._tokenizer.tokenize(
            text=evolved_instruction.instruction
        )
        with Pool(processes=num_cpus) as p:
            rouge_scores = p.map(
                partial(rouge_scorer._score_lcs, new_instruction_tokens),
                instruction_tokens,
            )
        rouge_scores = [score.fmeasure for score in rouge_scores]
        evolved_instruction.avg_similarity_score = np.mean(rouge_scores).item()
        evolved_instruction.most_similar_instructions = {
            instructions[i]: rouge_scores[i]
            for i in np.argsort(rouge_scores)[-10:][::-1]
        }

    # Get the corresponding outputs
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
