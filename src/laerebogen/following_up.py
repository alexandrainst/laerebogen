"""Creating follow-up queries to conversations."""

import json
import logging
import typing as t
from copy import deepcopy
from pathlib import Path

from pydantic import BaseModel, ValidationError
from tqdm.auto import tqdm

from .data_models import Conversation
from .vllm_utils import generate_text_with_vllm

if t.TYPE_CHECKING:
    from vllm import LLM

logger = logging.getLogger(__name__)


def add_follow_up_to_conversations(
    conversations: list[Conversation], prompt_path: str, model: "LLM"
) -> list[Conversation]:
    """Add follow-up queries to conversations using an instruction-tuned LLM.

    Args:
        conversations:
            The conversations to which follow-up queries will be added.
        prompt_path:
            Path to the prompt file containing the correction prompt.
        model:
            The instruction-tuned large language model to use for generating follow-up
            queries.

    Returns:
        The conversations with added follow-up queries.
    """
    # Load the tokenizer
    tokenizer = model.get_tokenizer()

    # Load the prompt
    with Path(prompt_path).open() as f:
        follow_up_prompt = f.read() + "\n"

    # Copy the conversations to avoid modifying the original ones
    extended_conversations = deepcopy(conversations)

    # Correct the instructions
    logger.info("Adding follow-up queries to conversations...")
    prompts = [
        follow_up_prompt.format(conversation=str(conversation))
        for conversation in conversations
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

    class ResponseFormat(BaseModel):
        """Response format for the vLLM model."""

        query: str
        output: str

    responses = generate_text_with_vllm(
        prompts=prompts, model=model, response_format=ResponseFormat
    )
    for conversation, response in zip(extended_conversations, responses):
        if response.done_reason == "stop":
            try:
                new_query = ResponseFormat.model_validate_json(
                    json_data=response.completion
                )
            except (json.JSONDecodeError, ValidationError):
                continue
            conversation.add_message(
                role="user",
                content=new_query.query.strip()
                if new_query.query.strip()
                else "<empty>",
            )
            conversation.add_message(
                role="assistant",
                content=new_query.output.strip()
                if new_query.output.strip()
                else "<empty>",
            )

    return extended_conversations
