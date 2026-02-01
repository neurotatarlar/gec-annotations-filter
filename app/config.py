"""Configuration defaults for the text cleaning and scoring pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class ProcessingConfig:
    """
    Configuration for text cleaning and filtering.

    The defaults follow the constraints provided for the Tatar GEC preparation task.
    """

    min_words: int = 6  # Drop texts shorter than this many words.
    max_words: int = 50  # Split or drop texts longer than this many words.
    min_chars: int = 30  # Drop texts shorter than this many characters.
    max_chars: int = 500  # Split or drop texts longer than this many characters.
    min_letters: int = 10  # Require at least this many letter characters total so texts contains not only emojis
    min_tatar_letters: int = 6  # Require at least this many Tatar-specific letters.
    cyrillic_only: bool = False  # If True, reject texts containing non-Cyrillic letters.
    tatar_letters: str = "ӘәҮүҖҗҢңӨөҺһ"  # Characters counted as Tatar-specific.
    system_messages: Tuple[str, ...] = (
        "Post is not in text format.",
        "Comment is not in text format.",
    )  # Exact strings considered system noise to drop.
    near_dup_distance: int = 2  # Default SimHash distance for near-duplicate removal.
    name_list_path: Path = Path(__file__).resolve().parents[1] / "data" / "tatar_names.json"  # Placeholder name replacements.
    gemini_keys_path: Path = Path(__file__).resolve().parents[1] / "data" / "gemini_keys.yaml"  # Location of Gemini API keys file.
    gemini_model: str = "models/gemini-3-flash-preview"  # Default Gemini model identifier.


DEFAULT_CONFIG = ProcessingConfig()
