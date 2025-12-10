import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import google.generativeai as genai
import yaml

DEFAULT_PROMPT = """You are an expert annotator of Tatar and Russian social media texts for a GEC (grammatical error correction) task.
Given ONE user message, reply ONLY with a compact JSON object with these keys:
{
  "main_language": "tatar" | "russian" | "mixed" | "other",
  "tatar_prob": float 0-1,
  "russian_prob": float 0-1,
  "russian_share": float 0-1,
  "error_density": "low" | "medium" | "high",
  "main_error_type": "none_or_minor" | "spelling" | "grammar_morphology" | "punctuation" | "word_order" | "lexical_choice",
  "meaning_clarity": float 0-1,
  "noise_score": float 0-1,
  "informality_level": "formal" | "neutral" | "informal" | "very_informal",
  "contains_hate_or_slurs": true | false,
  "contains_toxicity": "none" | "mild" | "strong",
  "non_fluent_prob": float 0-1,
  "overall_gec_usefulness": float 0-1,
  "explanation": "short reason in Russian or Tatar"
}
Constraints:
- Answer with JSON only, no prose, no markdown.
- If uncertain, guess but stay in allowed ranges/values.
Message:
"""


def load_keys(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"No Gemini keys file at {path}")
    data = yaml.safe_load(path.read_text())
    if not data:
        raise ValueError("Gemini keys file is empty")
    if isinstance(data, dict):
        keys = list(data.values())
    elif isinstance(data, list):
        keys = data
    else:
        raise ValueError("Gemini keys file must be a list or mapping")
    keys = [k.strip() for k in keys if k and str(k).strip()]
    if not keys:
        raise ValueError("Gemini keys file has no usable keys")
    return keys


class KeyRotator:
    """Round-robin over API keys with simple backoff."""

    def __init__(self, keys: List[str]):
        self.keys = keys
        self.idx = 0

    def next_key(self) -> str:
        key = self.keys[self.idx % len(self.keys)]
        self.idx = (self.idx + 1) % len(self.keys)
        return key


def configure_client(key: str, model: str):
    genai.configure(api_key=key)
    return genai.GenerativeModel(model)


def call_gemini(
    model: genai.GenerativeModel,
    prompt: str,
    text: str,
    retries: int = 2,
    cooldown: float = 0.5,
) -> str:
    """
    Call Gemini and return the raw text response. Caller should parse JSON.
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = model.generate_content(prompt + text, request_options={"timeout": 30})
            return resp.text or ""
        except Exception as exc:  # pragma: no cover - network/runtime
            last_err = exc
            time.sleep(cooldown * (attempt + 1))
    raise last_err  # type: ignore


def parse_json_response(resp_text: str) -> Dict:
    """
    Best-effort parse of Gemini response to JSON.
    """
    text = resp_text.strip()
    if not text:
        return {}
    # If model wraps JSON in markdown code fences.
    if text.startswith("```"):
        text = text.strip("` \n")
        # Remove optional json identifier
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return {}
    return {}
