"""Creating follow-up queries to conversations."""

import collections.abc as c
import logging
import typing as t
from pathlib import Path

import more_itertools as mit
from pydantic import ValidationError
from tqdm.auto import tqdm

from .data_models import Conversation, InstructionSample
from .vllm_utils import generate_text_with_vllm, load_vllm_model

if t.TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def add_follow_up_to_conversations(
    conversations: list[Conversation],
    prompt_path: str,
    model_id: str,
    num_follow_ups: int,
    batch_size: int,
) -> c.Generator[Conversation, None, None]:
    """Add follow-up queries to conversations using an instruction-tuned LLM.

    Args:
        conversations:
            The conversations to which follow-up queries will be added.
        prompt_path:
            Path to the prompt file containing the follow-up prompt.
        model_id:
            The ID of the instruction-tuned LLM to use for generating follow-up queries.
        num_follow_ups:
            The number of follow-up queries to add to each conversation.
        batch_size:
            The batch size to use for generating follow-up queries.

    Yields:
        The conversations with follow-up queries added.
    """
    with Path(prompt_path).open() as f:
        follow_up_prompt = f.read() + "\n"

    model = load_vllm_model(model_id=model_id)

    num_batches = len(conversations) // batch_size
    if len(conversations) % batch_size:
        num_batches += 1

    for batch in tqdm(
        iterable=mit.chunked(iterable=conversations, n=batch_size),
        desc="Adding follow-up queries",
        total=num_batches,
        unit="batch",
    ):
        extended_conversations: list[Conversation] = list()
        for _ in range(num_follow_ups):
            prompts = [
                follow_up_prompt.format(conversation=conversation.model_dump_json())
                for conversation in batch
            ]
            responses = generate_text_with_vllm(
                prompts=prompts,
                model=model,
                apply_chat_template=True,
                response_format=InstructionSample,
            )
            for conversation, response in zip(conversations, responses):
                if response.done_reason != "stop":
                    continue
                try:
                    new_query = InstructionSample.model_validate_json(
                        json_data=response.completion
                    )
                except ValidationError:
                    continue
                conversation.add_message(
                    role="user", content=new_query.instruction.strip()
                )
                conversation.add_message(
                    role="assistant", content=new_query.output.strip()
                )

        yield from extended_conversations
