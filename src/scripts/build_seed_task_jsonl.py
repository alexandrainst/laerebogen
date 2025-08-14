"""Convert raw CSV seed task spreadsheets to a single JSONL file.

Usage:
    python build_seed_task_jsonl.py \
        --alpaca-path <alpaca_path> \
        --trustllm-path <trustllm_path> \
        [--output-dir <output_path>]
"""

import json
import logging
import warnings
from pathlib import Path

import click
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("build_seed_task_jsonl")


@click.command()
@click.option(
    "--alpaca-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    required=True,
    help="Path to the Alpaca seed tasks CSV file.",
)
@click.option(
    "--trustllm-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    required=True,
    help="Path to the TrustLLM seed tasks CSV file.",
)
@click.option(
    "--output-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    default="data",
    show_default=True,
    help="Directory to save the output JSONL file.",
)
def main(alpaca_path: str, trustllm_path: str, output_dir: str) -> None:
    """Convert raw CSV seed task spreadsheets to a single JSONL file."""
    # Process the Alpaca data
    alpaca_df = pd.read_csv(alpaca_path)
    alpaca_records = process_alpaca_data(df=alpaca_df)

    # Process the TrustLLM data
    trustllm_df = pd.read_csv(trustllm_path)
    trustllm_records = process_trustllm_data(df=trustllm_df)

    # Combine and store the processed data
    all_records = alpaca_records + trustllm_records
    output_path = Path(output_dir, "seed_tasks.jsonl")
    with output_path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(all_records):
            json_record = json.dumps(record, ensure_ascii=False)
            include_newline = idx < len(all_records) - 1
            f.write(f"{json_record}\n" if include_newline else json_record)

    logger.info(
        f"Built seed tasks JSONL file with {len(all_records):,} records "
        f"and saved it to {output_path.resolve()!s}."
    )


def process_alpaca_data(df: pd.DataFrame) -> list[dict[str, str]]:
    """Process Alpaca seed tasks data.

    Args:
        df:
            DataFrame containing Alpaca seed tasks data.

    Returns:
        List of dictionaries with processed seed tasks.
    """
    # Only keep the rows where the task was corrected
    df = df.query("corrected == 'yes'")

    # If some of the translated columns were corrected, then we replace the original
    # translated columns with the corrected ones
    df.instruction = df.apply(
        lambda row: row.corrected_instruction
        if pd.notna(row.corrected_instruction)
        else row.instruction,
        axis=1,
    )
    df.input = df.apply(
        lambda row: row.corrected_input if pd.notna(row.corrected_input) else row.input,
        axis=1,
    )
    df.output = df.apply(
        lambda row: row.corrected_output
        if pd.notna(row.corrected_output)
        else row.output,
        axis=1,
    )

    # Convert NaNs to empty strings
    df.fillna("", inplace=True)

    # Set up the columns to match the Alpaca format
    df["id"] = [f"alpaca_task_{i}" for i in range(len(df))]
    df["instances"] = df.apply(
        lambda row: [dict(input=row.input, output=row.output)], axis=1
    )
    df["is_classification"] = None
    df["metadata"] = df.apply(
        lambda row: dict(
            source="alpaca-da", has_input=bool(row.input), original_source="alpaca"
        ),
        axis=1,
    )

    # Only keep the relevant columns
    df = df[["id", "name", "instruction", "instances", "is_classification", "metadata"]]

    # Convert the DataFrame to a list of dictionaries
    records = df.to_dict(orient="records")

    return records


def process_trustllm_data(df: pd.DataFrame) -> list[dict[str, str]]:
    """Process TrustLLM seed tasks data.

    Args:
        df:
            DataFrame containing TrustLLM seed tasks data.

    Returns:
        List of dictionaries with processed seed tasks.
    """
    # Only keep the rows where the task is useful and was corrected
    df = df.query("corrected == 'yes' and useful == 'yes'")

    # Rename "completion" to "output" to match the Alpaca format
    df.rename(
        columns=dict(completion="output", corrected_completion="corrected_output"),
        inplace=True,
    )

    # If some of the translated columns were corrected, then we replace the original
    # translated columns with the corrected ones
    df.instruction = df.apply(
        lambda row: row.corrected_instruction
        if pd.notna(row.corrected_instruction)
        else row.instruction,
        axis=1,
    )
    df.output = df.apply(
        lambda row: row.corrected_output
        if pd.notna(row.corrected_output)
        else row.output,
        axis=1,
    )

    # Set up the columns to match the Alpaca format
    df["id"] = [f"trustllm_task_{i}" for i in range(len(df))]
    df["name"] = [f"trustllm_{i}" for i in range(len(df))]
    df["instances"] = df.apply(lambda row: [dict(input="", output=row.output)], axis=1)
    df["is_classification"] = None
    df["metadata"] = [
        dict(source="trustllm", has_input=False, original_source="TrustLLM")
        for _ in range(len(df))
    ]

    # Only keep the relevant columns
    df = df[["id", "name", "instruction", "instances", "is_classification", "metadata"]]

    # Convert the DataFrame to a list of dictionaries
    records = df.to_dict(orient="records")

    return records


if __name__ == "__main__":
    main()
