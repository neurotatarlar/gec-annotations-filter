from pathlib import Path
from typing import Optional

import typer

from .commands import (
    ingest_cmd,
    dedup_cmd,
    toxicity_cmd,
    gemini_cmd,
    export_parquet_cmd,
    prepare_import_cmd,
)
from .config import DEFAULT_CONFIG

app = typer.Typer(add_completion=False, help="Pipeline to prepare texts for Tatar GEC annotation.")


@app.command()
def ingest(
    db_path: Path = typer.Option(Path("data/pipeline.sqlite"), "--db-path", help="Path to sqlite database for intermediate data."),
    dataset: str = typer.Option("yasalma/vk-messages", help="HF dataset identifier or local path."),
    split: str = typer.Option("train", help="Dataset split to stream."),
    cache_dir: Path = typer.Option(Path("data/hf_cache"), help="Local cache dir for streamed dataset shards."),
    text_field: str = typer.Option("text", help="Field containing the raw message."),
    url_field: str = typer.Option("url", help="Field containing the message URL."),
    limit: Optional[int] = typer.Option(None, help="Optional limit for debugging."),
    batch_size: int = typer.Option(500, help="Number of cleaned segments to insert per transaction."),
    report_path: Optional[Path] = typer.Option(Path("reports/ingest_report.json"), help="Write JSON counters about filter reasons."),
    output_table: str = typer.Option("clean_segments", "--output-table", help="Destination table to write cleaned segments."),
):
    ingest_cmd(db_path, dataset, split, cache_dir, text_field, url_field, limit, batch_size, report_path, output_table)


@app.command()
def dedup(
    db_path: Path = typer.Option(Path("data/pipeline.sqlite"), "--db-path", help="Existing sqlite database."),
    source_table: str = typer.Option("clean_segments", help="Table to read from."),
    output_table: str = typer.Option("dedup_segments", "--output-table", help="Destination table for deduplicated rows."),
    distance: int = typer.Option(DEFAULT_CONFIG.near_dup_distance, help="SimHash Hamming distance for near-duplicate removal."),
    short_distance: int = typer.Option(1, help="Stricter SimHash distance for short texts."),
    short_token_threshold: int = typer.Option(8, help="Token count threshold for short text handling."),
    batch_size: int = typer.Option(1000, help="Rows to process per batch."),
    report_path: Optional[Path] = typer.Option(Path("reports/dedup_report.json"), help="Write JSON counters for dedup stage."),
):
    dedup_cmd(db_path, source_table, output_table, distance, short_distance, short_token_threshold, batch_size, report_path)


@app.command()
def toxicity(
    db_path: Path = typer.Option(Path("data/pipeline.sqlite"), "--db-path", help="Existing sqlite database."),
    source_table: str = typer.Option("dedup_segments", help="Table to read from."),
    output_table: str = typer.Option("toxicity_segments", "--output-table", help="Destination table for toxicity-scored rows."),
    threshold: Optional[float] = typer.Option(None, help="Optional threshold: drop texts with toxicity_score >= threshold."),
    batch_size: int = typer.Option(32, help="Batch size for the model."),
    device: int = typer.Option(-1, help="Transformers device id (-1 = CPU)."),
    report_path: Optional[Path] = typer.Option(Path("reports/toxicity_report.json"), help="Write JSON counters for toxicity stage."),
):
    toxicity_cmd(db_path, source_table, output_table, threshold, batch_size, device, report_path)


@app.command("gemini")
def gemini_score(
    db_path: Path = typer.Option(Path("data/pipeline.sqlite"), "--db-path", help="Existing sqlite database."),
    source_table: str = typer.Option("toxicity_segments", help="Table to read from."),
    output_table: str = typer.Option("llm_segments", "--output-table", help="Destination table for LLM-scored rows."),
    keys_path: Path = typer.Option(DEFAULT_CONFIG.gemini_keys_path, help="YAML file containing Gemini API keys (list or mapping)."),
    model: str = typer.Option(DEFAULT_CONFIG.gemini_model, help="Gemini model name."),
    prompt_path: Optional[Path] = typer.Option(None, help="Optional path to custom prompt; defaults to built-in."),
    batch_size: int = typer.Option(64, help="Rows to read from DB per loop and send per Gemini request."),
    max_rows: Optional[int] = typer.Option(None, help="Optional limit for debugging."),
    report_path: Optional[Path] = typer.Option(Path("reports/gemini_report.json"), help="Write JSON counters for Gemini stage."),
):
    gemini_cmd(db_path, source_table, output_table, keys_path, model, prompt_path, batch_size, max_rows, report_path)


@app.command()
def export_parquet(
    db_path: Path = typer.Option(Path("data/pipeline.sqlite"), "--db-path", help="Existing sqlite database."),
    output: Path = typer.Argument(..., help="Destination parquet file."),
    table: str = typer.Option(
        "toxicity_segments",
        help="Table to export. Will fall back to dedup_segments if empty.",
    ),
):
    export_parquet_cmd(db_path, output, table)


@app.command("prepare-import")
def prepare_import(
    db_path: Path = typer.Option(Path("data/pipeline.sqlite"), "--db-path", help="Existing sqlite database."),
    source_table: str = typer.Option("llm_segments", help="Gemini-scored table to read from."),
    output: Path = typer.Option('data/import.json',  help="Destination JSON file containing array of {{id, text}}."),
    limit: Optional[int] = typer.Option(None, help="Optional maximum number of items to include."),
    extreme_share: float = typer.Option(0.15, help="Share (0-0.5) for low/high error_density buckets."),
    dry_run: bool = typer.Option(False, help="If true, just report counts without writing or marking exported."),
    keep_proportions: bool = typer.Option(True, help="If true, preserve low/medium/high proportions even if total is smaller."),
):
    prepare_import_cmd(db_path, source_table, output, limit, extreme_share, dry_run, keep_proportions)


if __name__ == "__main__":
    app()
