"""Finetune a pre-trained generative language model on the generated dataset."""

import importlib.util
import logging
import os
import typing as t
import warnings

if importlib.util.find_spec("unsloth") is not None or t.TYPE_CHECKING:
    from unsloth import FastLanguageModel

import torch
import wandb
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from transformers import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import IntervalStrategy, SchedulerType
from transformers.training_args import OptimizerNames
from trl import SFTConfig, SFTTrainer, setup_chat_format

logger = logging.getLogger(__package__)

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"


def finetune_model(
    base_model_id: str,
    new_model_id: str,
    val_samples: int,
    load_in_4bit: bool,
    max_seq_length: int,
    lora_rank: t.Literal[8, 16, 32, 64, 128, 256, 512],
    learning_rate: float,
    weight_decay: float,
    neftune_noise_alpha: int,
    per_device_batch_size: int,
    total_batch_size: int,
    num_epochs: int,
    warmup_ratio: float,
    logging_steps: int,
    eval_steps: int,
    dataloader_num_workers: int,
    use_wandb: bool,
    testing: bool,
) -> None:
    """Finetune a pre-trained generative language model on a dataset.

    Args:
        base_model_id:
            The model ID of the pre-trained model to finetune. Needs to exist on the
            Unsloth Hugging Face Hub organisation (hf.co/unsloth). You can leave out the
            organisation prefix, e.g., "gemma-2-9b" instead of "unsloth/gemma-2-9b".
        new_model_id:
            The model ID of the finetuned model. The model will be uploaded to the
            Hugging Face Hub with the ID `alexandrainst/{new_model_id}`.
        val_samples:
            The number of samples to use for the validation set. The rest of the
            samples will be used for the training set.
        load_in_4bit:
            Whether to load the model in 4-bit mode. If `True`, the model will be
            loaded in 4-bit mode, which will reduce the memory usage of the model, but
            may also reduce the performance of the model. If `False`, the model will be
            loaded in 16-bit mode, which will increase the memory usage of the model,
            but may also increase the performance of the model.
        max_seq_length:
            The maximum sequence length of the model. RoPE scaling is implemented, so
            the model will be able to handle larger sequence lengths in any case.
        lora_rank:
            The rank of the LoRA adapter to use. The rank determines the number of
            parameters in the adapter, and thus the capacity of the adapter to model
            the dataset. The rank should be chosen based on the size of the dataset and
            the complexity of the task. A higher rank will allow the adapter to model
            more complex patterns in the dataset, but will also require more data to
            train effectively. Can be one of 8, 16, 32, 64, or 128.
        learning_rate:
            The maximum learning rate to use for training. The learning rate will
            increase linearly from 0 to this value over the warmup steps.
        weight_decay:
            The weight decay to use as part of the AdamW optimiser.
        neftune_noise_alpha:
            The alpha parameter to use for Neftune noise. This parameter determines the
            amount of noise to add to the logits during training. A higher value will
            add more noise to the logits, which can help to regularise the model and
            improve generalisation. A lower value will add less noise to the logits,
            which can help to improve the performance of the model on the training set.
        per_device_batch_size:
            The batch size to use per device during training. Must divide
            `total_batch_size`.
        total_batch_size:
            The total batch size to use for each gradient update. We will be using
            gradient accumulation to ensure that we accumulate enough gradients to
            update the model parameters in each update. Must be divisible by
            `per_device_batch_size`.
        num_epochs:
            The number of epochs to train the model for. An epoch is one pass through
            the entire dataset.
        warmup_ratio:
            The ratio of the total number of steps to use for warmup. The learning rate
            will increase linearly from 0 to `learning_rate` over the warmup steps.
        logging_steps:
            How often logging should occur during training.
        eval_steps:
            How often evaluation should occur during training.
        dataloader_num_workers:
            The number of workers to use for loading the data into the dataloader. This
            can speed up data loading, but may also cause issues with some datasets.
        use_wandb:
            Whether to use Weights & Biases for logging. This requires that the Weights
            and Biases API key is set in the environment variable `WANDB_API_KEY`.
        testing:
            Whether to use a small subset of the dataset for testing. This can be useful
            for debugging and development, but may not provide a good estimate of the
            model's performance on the full dataset.
    """
    assert total_batch_size % per_device_batch_size == 0, (
        f"Total batch size ({total_batch_size}) must be divisible by per device batch "
        f"size ({per_device_batch_size})."
    )

    match base_model_id.count("/"):
        case 0:
            base_model_id = f"unsloth/{base_model_id}"
        case 1:
            organisation = base_model_id.split("/")[0]
            assert organisation == "unsloth", (
                "Organisation of base model ID must be 'unsloth', but got "
                f"{organisation!r}."
            )
        case _:
            raise ValueError(
                "Base model ID must not contain more than one forward slash."
            )

    if use_wandb and "WANDB_API_KEY" not in os.environ:
        raise ValueError(
            "Weights & Biases API key not set in environment variables. Please set "
            "the `WANDB_API_KEY` environment variable to use Weights & Biases, or "
            "run the script with `use_wandb=false` to disable Weights & Biases."
        )

    if testing:
        total_batch_size = 2
        per_device_batch_size = 1
        logging_steps = 1
        use_wandb = False
        val_samples = 2
        eval_steps = 5
        num_epochs = 10
        logger.info("Running in testing mode.")

    logger.info("Loading the dataset...")
    dataset = load_dataset(
        path="alexandrainst/laerebogen",
        split="train",
        token=os.getenv("HUGGINGFACE_API_KEY", True),
    )
    assert isinstance(dataset, Dataset)

    # Remove empty samples
    dataset = dataset.filter(
        function=lambda x: len(x["messages"]) > 0
        and all(
            len(message["content"]) > 0
            for message in x["messages"]
            if message["role"] in {"user", "assistant"}
        ),
        desc="Filtering empty samples",
    )

    if testing:
        dataset = dataset.select(range(4 + val_samples))

    logger.info(f"Loaded dataset with {len(dataset):,} examples.")

    dataset = dataset.train_test_split(test_size=val_samples, seed=4242, shuffle=True)
    logger.info(
        f"Using {len(dataset['train']):,} samples for training and "
        f"{len(dataset['test']):,} samples for validation."
    )

    logger.info(f"Loading model and tokenizer with ID {base_model_id}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_id,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        token=os.getenv("HUGGINGFACE_API_KEY", True),
    )
    model, tokenizer = setup_chat_format(
        model=model, tokenizer=tokenizer, format="chatml"
    )
    tokenizer.padding_side = "left"
    assert isinstance(model, PreTrainedModel)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    if not testing:
        logger.info(
            f"Pushing the tokenizer to the Hugging Face Hub with ID {new_model_id}..."
        )
        tokenizer.push_to_hub(
            repo_id=f"alexandrainst/{new_model_id}",
            token=os.getenv("HUGGINGFACE_API_KEY", True),
            private=True,
        )

    logger.info("Converting the model to a PEFT model...")
    peft_model = FastLanguageModel.get_peft_model(
        model=model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_rank * 2,
        lora_dropout=0,
        bias="none",
        use_rslora=True,
        use_gradient_checkpointing=True,
        random_state=4242,
    )

    num_devices = torch.cuda.device_count()
    gradient_accumulation_steps = (
        total_batch_size // per_device_batch_size // num_devices
    )

    sft_config = SFTConfig(
        output_dir="outputs",
        run_name=new_model_id,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        eval_strategy=IntervalStrategy.STEPS,
        eval_steps=eval_steps,
        optim=OptimizerNames.PAGED_ADAMW_8BIT,
        weight_decay=weight_decay,
        lr_scheduler_type=SchedulerType.COSINE,
        seed=4242,
        report_to=["wandb"] if use_wandb else [],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        hub_model_id=new_model_id,
        push_to_hub=False,
        dataloader_num_workers=dataloader_num_workers,
        max_length=max_seq_length,
        neftune_noise_alpha=neftune_noise_alpha,
        assistant_only_loss=True,
    )

    def formatting_func(examples: dict) -> list[str]:
        """Format the example for training.

        Args:
            examples:
                A batch of examples from the dataset.

        Returns:
            A list of formatted examples, ready for training.
        """
        formatted_texts = tokenizer.apply_chat_template(
            conversation=examples["messages"],
            add_generation_prompt=True,
            tokenize=False,
        )
        if isinstance(formatted_texts, str):
            formatted_texts = [formatted_texts]
        assert isinstance(formatted_texts, list)
        return formatted_texts

    logger.info("Creating the SFT trainer...")
    trainer = SFTTrainer(
        model=peft_model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=sft_config,
        formatting_func=formatting_func,
    )

    if use_wandb and not testing:
        wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb.init(
            project="laerebogen-finetuning",
            config=dict(
                base_model_id=base_model_id,
                new_model_id=new_model_id,
                load_in_4bit=load_in_4bit,
                max_seq_length=max_seq_length,
                lora_rank=lora_rank,
                total_batch_size=total_batch_size,
                train_samples=len(dataset["train"]),
                val_samples=len(dataset["test"]),
            )
            | sft_config.to_dict(),
        )

    logger.info("Training the model...")
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=UserWarning)
        trainer.train()

    logger.info("Finetuning complete. Generating a sample response...")
    peft_model = FastLanguageModel.for_inference(peft_model)
    if testing:
        doc = dataset["train"][0]["messages"][0]["content"]
    else:
        doc = "Hvad synes du om Danish Foundation Models projektet?"
    input_ids = tokenizer.apply_chat_template(
        conversation=[dict(role="user", content=doc)],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    assert isinstance(input_ids, torch.Tensor)
    input_ids = input_ids.to("cuda")
    outputs = peft_model.generate(input_ids=input_ids, max_new_tokens=256)
    response = tokenizer.decode(outputs[0, input_ids.size(1) :])
    logger.info("*** Sample response ***")
    logger.info(f"Input: {doc!r}")
    logger.info(f"Response: {response!r}")

    if not testing:
        logger.info(
            f"Pushing the model to the Hugging Face Hub with ID {new_model_id}..."
        )
        peft_model.push_to_hub_merged(
            repo_id=f"alexandrainst/{new_model_id}",
            tokenizer=tokenizer,
            save_method="merged_16bit",
            token=os.getenv("HUGGINGFACE_API_KEY", True),
            private=True,
        )

    if use_wandb:
        wandb.finish()
