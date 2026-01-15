"""Finetune a pre-trained generative language model on the generated dataset."""

import logging
import os
import random
import warnings
from functools import partial

import numpy as np
import torch
import wandb
from datasets import Dataset, load_dataset
from datasets.dataset_dict import DatasetDict
from dotenv import load_dotenv
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer_utils import EvalPrediction, IntervalStrategy, SchedulerType
from transformers.training_args import OptimizerNames
from trl import SFTConfig, SFTTrainer, clone_chat_template

logger = logging.getLogger(__package__)

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"


def finetune_model(
    base_model_id: str,
    new_model_id: str,
    dataset_id: str,
    val_samples: int,
    learning_rate: float,
    weight_decay: float,
    neftune_noise_alpha: int,
    per_device_batch_size: int,
    total_batch_size: int,
    num_epochs: int,
    warmup_ratio: float,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
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
            Hugging Face Hub with this ID.
        dataset_id:
            The dataset to finetune the model on. This must be the ID of a dataset on
            the Hugging Face Hub, and must contain a `messages` feature containing lists
            of dictionaries, representing conversations.
        val_samples:
            The number of samples to use for the validation set. The rest of the
            samples will be used for the training set.
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
        save_steps:
            How often the model should be saved during training.
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
        f"Total batch size ({total_batch_size}) must be divisible by per "
        f"device batch size ({per_device_batch_size})."
    )

    if use_wandb and "WANDB_API_KEY" not in os.environ:
        raise ValueError(
            "Weights & Biases API key not set in environment variables. Please set "
            "the `WANDB_API_KEY` environment variable to use Weights & Biases, or "
            "run the script with `use_wandb=false` to disable Weights & Biases."
        )

    # Note if we're on the main process, if we are running in a distributed setting
    is_main_process = os.getenv("RANK", "0") == "0"

    # Load the dataset from the Hugging Face Hub
    if is_main_process:
        logger.info(f"Loading dataset with ID {dataset_id}...")
    dataset_dict = load_dataset(dataset_id, use_auth_token=os.getenv("HF_API_TOKEN"))
    assert isinstance(dataset_dict, DatasetDict), (
        f"Expected dataset_dict to be of type DatasetDict, got {type(dataset_dict)}"
    )
    assert "train" in dataset_dict, (
        f"Expected 'train' split in dataset, got {dataset_dict.keys()}"
    )
    dataset = dataset_dict["train"]
    assert isinstance(dataset, Dataset), (
        f"Expected dataset to be of type Dataset, got {type(dataset)}"
    )
    assert "messages" in dataset.column_names, (
        f"Expected 'messages' column in dataset, got {dataset.column_names}"
    )

    if testing:
        total_batch_size = min(2, total_batch_size)
        per_device_batch_size = 1
        logging_steps = 1
        use_wandb = False
        val_samples = min(2, val_samples)
        eval_steps = min(5, eval_steps)
        num_epochs = 10
        if is_main_process:
            logger.info("Running in testing mode.")

        dataset = dataset.take(n=4 + val_samples)
        assert isinstance(dataset, Dataset), (
            f"Expected dataset to be of type Dataset, got {type(dataset)}"
        )

    if is_main_process:
        logger.info("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, token=os.getenv("HF_API_TOKEN", True)
    )
    assert isinstance(tokenizer, PreTrainedTokenizerFast), (
        "Expected tokenizer to be of type PreTrainedTokenizerFast, got "
        f"{type(tokenizer)}"
    )

    if is_main_process:
        logger.info("Loading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, token=os.getenv("HF_API_TOKEN", True)
    )
    assert isinstance(model, PreTrainedModel), (
        f"Expected model to be of type PreTrainedModel, got {type(model)}"
    )

    if is_main_process:
        logger.info("Setting up chat template...")

    model, tokenizer, tokens_added = clone_chat_template(
        model=model,
        tokenizer=tokenizer,  # pyrefly: ignore[bad-argument-type]
        source_tokenizer_path="danish-foundation-models/Meta-Llama-3.1-8B-laerebogen",
    )

    if is_main_process:
        logger.info("Tokenizing the dataset...")

    def tokenize_function(examples: dict, tokenizer: "PreTrainedTokenizerFast") -> dict:
        """Tokenize a batch of examples.

        Args:
            examples:
                A batch of examples containing `messages`.

        Returns:
            A batch of tokenized examples.
        """
        documents = tokenizer.apply_chat_template(examples["messages"], tokenize=False)
        model_inputs = tokenizer(documents)
        return model_inputs

    mapped = dataset.map(
        partial(tokenize_function, tokenizer=tokenizer), batched=True, batch_size=16
    )
    assert isinstance(mapped, Dataset), (
        f"Expected mapped to be of type Dataset, got {type(mapped)}"
    )
    dataset = mapped

    if is_main_process:
        logger.info(
            "Splitting dataset into train and validation sets with "
            f"{val_samples} validation samples..."
        )
    train_val_split = dataset.train_test_split(test_size=val_samples, seed=42)
    train_split = train_val_split["train"]
    val_split = train_val_split["test"]
    assert isinstance(train_split, Dataset), (
        f"Expected train_split to be of type Dataset, got {type(train_split)}"
    )
    assert isinstance(val_split, Dataset), (
        f"Expected val_split to be of type Dataset, got {type(val_split)}"
    )

    num_devices = torch.cuda.device_count()
    gradient_accumulation_steps = (
        1 if testing else (total_batch_size // per_device_batch_size // num_devices)
    )

    training_args = SFTConfig(
        output_dir="outputs",
        run_name=new_model_id,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_accumulation_steps=1,
        num_train_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        eval_strategy=IntervalStrategy.STEPS,
        eval_steps=eval_steps,
        save_strategy=IntervalStrategy.NO if testing else IntervalStrategy.STEPS,
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True if not testing else False,
        optim=OptimizerNames.PAGED_ADAMW_8BIT,
        weight_decay=weight_decay,
        lr_scheduler_type=SchedulerType.COSINE,
        seed=4242,
        report_to=["wandb"] if use_wandb and is_main_process else [],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        hub_model_id=new_model_id,
        push_to_hub=False,
        hub_private_repo=True,
        dataloader_num_workers=dataloader_num_workers,
        neftune_noise_alpha=neftune_noise_alpha,
        hub_token=os.getenv("HF_API_TOKEN"),
        gradient_checkpointing_kwargs=dict(use_reentrant=False),
        eos_token="<|end_of_text|>",
    )

    if is_main_process:
        logger.info(f"Loading base model and tokenizer with ID {base_model_id}...")

    if is_main_process:
        logger.info("Creating the trainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_split,
        eval_dataset=val_split,
        args=training_args,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer),
    )

    if use_wandb and is_main_process and not testing:
        wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb.init(
            project="job-bot",
            config=dict(
                base_model_id=base_model_id,
                new_model_id=new_model_id,
                total_batch_size=total_batch_size,
                val_samples=val_samples,
            )
            | training_args.to_dict(),
        )

    if is_main_process:
        logger.info("Training the model...")
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=UserWarning)
        trainer.train()

    if not testing and is_main_process:
        logger.info(
            f"Pushing the model to the Hugging Face Hub with ID {new_model_id}..."
        )
        trainer.push_to_hub()

    if use_wandb and is_main_process:
        wandb.finish()


def compute_metrics(
    eval_pred: EvalPrediction, tokenizer: "PreTrainedTokenizerFast"
) -> dict[str, float]:
    """Compute metrics for the evaluation.

    Args:
        eval_pred:
            The evaluation prediction object containing the predictions and labels.
        tokenizer:
            The tokenizer used for decoding the predictions.

    Returns:
        A dictionary containing the computed metrics.
    """
    # Note if we're on the main process, if we are running in a distributed setting
    is_main_process = os.getenv("RANK", "0") == "0"

    # Get the labels
    label_ids = eval_pred.label_ids
    label_ids = [labels[labels != -100].tolist() for labels in label_ids]
    labels = tokenizer.batch_decode(sequences=label_ids, skip_special_tokens=True)

    # Get the predictions
    logits = (
        eval_pred.predictions[0]
        if isinstance(eval_pred.predictions, tuple)
        else eval_pred.predictions
    )
    assert isinstance(logits, np.ndarray), (
        f"Expected logits to be of type np.ndarray, got {type(logits)}: {logits}"
    )
    predictions = np.argmax(logits, axis=-1)
    completions = tokenizer.batch_decode(
        sequences=predictions, skip_special_tokens=True
    )

    # Log example prediction
    if is_main_process:
        random_idx = random.randint(0, len(completions) - 1)
        logger.info(f"Example prediction:\n{completions[random_idx]!r}")
        logger.info(f"Associated labels:\n{labels[random_idx]!r}")

    return dict()
