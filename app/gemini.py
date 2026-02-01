"""Gemini prompt templates, schema validation, and request helpers."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Any

from pydantic import BaseModel, TypeAdapter

from google.genai import types
import yaml

DEFAULT_PROMPT = """
You are a labeling model for preparing training data for a Grammar Error Correction (GEC) system for the Tatar language.

Your job:
- Receive a JSON object with multiple short texts (mostly Tatar, with possible Russian words and code-switching).
- For EACH text, output a JSON object with numeric scores and categorical labels that describe:
  - language mix,
  - amount of grammatical/spelling errors,
  - clarity/quality,
  - style/informality,
  - usefulness as a GEC training example.

GENERAL RULES
-------------
- Work at the LEVEL OF EACH ITEM INDEPENDENTLY. Do not compare items.
- Be CONSISTENT, not perfect. These are heuristic labels to help data selection.
- For probabilities and shares:
  - Use numbers from 0.0 to 1.0.
  - Use a dot as decimal separator.
  - Up to 3 decimal places is enough (e.g. 0.0, 0.25, 0.733).
- Always return VALID JSON with DOUBLE QUOTES for all keys and string values.
- Do NOT include any comments, explanations, or extra keys.

INPUT FORMAT
------------
You will receive JSON like:

{
  "task": "tatar_gec_scoring",
  "items": [
    {
      "id": "m1",
      "text": "TEXT 1 HERE"
    },
    {
      "id": "m2",
      "text": "TEXT 2 HERE"
    }
  ]
}

Each item:
- "id": arbitrary string identifier (you must copy it back unchanged).
- "text": one short message, usually from a social network.

OUTPUT FORMAT
-------------
You MUST output a single JSON object with the following shape:

[
    {
      "id": "<same id as input>",
      "labels": {
        "main_language": "tatar | russian | mixed | other",
        "tatar_prob": 0.0,
        "russian_share": 0.0,
        "error_share": 0.0,
        "error_density": "low | medium | high",
        "main_error_type": "none_or_minor | spelling | grammar_morphology | punctuation | word_order | lexical_choice",
        "non_fluent_prob": 0.0,
        "meaning_clarity": 0.0,
        "noise_score": 0.0,
        "overall_gec_usefulness": 0.0
      }
    }
]

- "id": MUST exactly match the input "id".
- "labels": an object with exactly the keys above.

DETAILED FIELD DEFINITIONS
---------------------------

LANGUAGE & CODE-SWITCHING
- "main_language":
  - "tatar"  = mostly Tatar.
  - "russian" = mostly Russian.
  - "mixed"   = heavy code-switching between Tatar and Russian (or other languages).
  - "other"   = neither Tatar nor Russian as main language.
- "tatar_prob": Probability (0–1) that the main language is Tatar.
- "russian_share": Fraction (0–1) of words that are clearly Russian (or Russian-based), including Russian words inside an otherwise Tatar sentence.

ERROR SIGNALS
-------------
- "error_share":
  - Approximate fraction (0–1) of tokens in the text that contain a grammatical or spelling error.
  - 0.0 = essentially error-free; 1.0 = almost every word has some error.
- "error_density":
  - "low"    = 0–2 small mistakes, mostly correct.
  - "medium" = noticeable number of errors, but still readable.
  - "high"   = many errors, very errorful text.
- "main_error_type":
  - "none_or_minor"       = text is essentially correct or has only tiny issues.
  - "spelling"            = main issues are typos, orthography, character mistakes.
  - "grammar_morphology"  = case endings, suffixes, agreement, verb forms, etc.
  - "punctuation"         = commas, periods, question marks, etc.
  - "word_order"          = unnatural or incorrect word order.
  - "lexical_choice"      = wrong or unnatural word choice, calques from Russian, etc.
- "non_fluent_prob":
  - Probability (0–1) that the text looks like it was written by a non-fluent or strongly non-standard writer of Tatar.
  - Consider big grammar mistakes, strange morphology, wrong word choice, etc.
  - This is a heuristic, not a real psychological measure.

READABILITY / QUALITY
---------------------
- "meaning_clarity":
  - 0.0 = the message is essentially impossible to understand for a fluent Tatar speaker.
  - 1.0 = very clear meaning, even if there are errors.
- "noise_score":
  - 0.0 = normal human text.
  - 1.0 = mostly noise, spam, random characters, repeated symbols, etc.

OVERALL USEFULNESS FOR GEC
--------------------------
- "overall_gec_usefulness":
  - Single score 0.0–1.0:
  - 0.0 = very bad example for GEC training (nonsense, unreadable, or not Tatar at all).
  - 1.0 = excellent training example (clearly Tatar, understandable, with some errors that a GEC model should learn to correct).
  - Consider:
    - Is it mainly Tatar?
    - Is the meaning clear?
    - Are there some errors (not 0, not total chaos)?
    - Is it a realistic example of social-media Tatar?

EXAMPLES
--------
Example:

Input item:
[
  {
    "id": "1f6a3d81442222a989095eefa570f5c6381ee5a4#0",
    "text": "Мин бүген мәктәпкә барам."
  },
  {
    "id": "a1d37388534e91c18942f8f5e9e623c9622ac944#0",
    "text": "Мин сегодня в школга барам, капец уже опаздывам""
  },
]

Expected labels (illustrative only):
[
  {
    "id": "1f6a3d81442222a989095eefa570f5c6381ee5a4#0",
    "labels": {
      "main_language": "tatar",
      "tatar_prob": 0.98,
      "russian_share": 0.0,
      "error_share": 0.0,
      "error_density": "low",
      "main_error_type": "none_or_minor",
      "non_fluent_prob": 0.05,
      "meaning_clarity": 1.0,
      "noise_score": 0.0,
      "overall_gec_usefulness": 0.6
    }
  },
  {
    "id": "a1d37388534e91c18942f8f5e9e623c9622ac944#0",
    "labels": {
      "main_language": "mixed",
      "tatar_prob": 0.7,
      "russian_share": 0.4,
      "error_share": 0.5,
      "error_density": "high",
      "main_error_type": "grammar_morphology",
      "non_fluent_prob": 0.7,
      "meaning_clarity": 0.9,
      "noise_score": 0.1,
      "overall_gec_usefulness": 0.85
    }
  }
]

NOW YOUR TASK
-------------
1. Read the JSON input below.
2. For EACH item in "items":
   - Analyze the text.
   - Produce labels according to the definitions above.
3. Output a SINGLE JSON object with the structure:

[
    {
      "id": "...",
      "labels": { ... }
    },
    ...
]

4. Do NOT output anything except this JSON.

INPUT:
"""


def load_keys(path: Path) -> List[str]:
    """Load Gemini API keys from a YAML list or mapping."""
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


ALLOWED_LANGS = {"tatar", "russian", "mixed", "other"}
ALLOWED_ERROR_DENSITY = {"low", "medium", "high"}
ALLOWED_ERROR_TYPE = {
    "none_or_minor",
    "spelling",
    "grammar_morphology",
    "punctuation",
    "word_order",
    "lexical_choice",
}


class LabelPayload(BaseModel):
    """Validated label payload returned by the scoring model."""

    main_language: str
    tatar_prob: float
    russian_share: float
    error_share: float
    error_density: str
    main_error_type: str
    non_fluent_prob: float
    meaning_clarity: float
    noise_score: float
    overall_gec_usefulness: float


class ScoredItem(BaseModel):
    """Single scored item with id and labels."""

    id: str
    labels: LabelPayload


def _coerce_float(val: Any, field: str) -> float:
    """Coerce a value to float with a helpful error message."""
    try:
        return float(val)
    except Exception as e:
        raise ValueError(f"{field} must be a float") from e


def _validate_labels(ldict: Dict[str, Any]) -> LabelPayload:
    """Validate label fields and normalize enums."""
    if not isinstance(ldict, dict):
        raise ValueError("labels must be an object")
    ml = str(ldict.get("main_language", "")).lower()
    if ml not in ALLOWED_LANGS:
        raise ValueError(f"main_language invalid: {ml}")
    ed = str(ldict.get("error_density", "")).lower()
    if ed not in ALLOWED_ERROR_DENSITY:
        raise ValueError(f"error_density invalid: {ed}")
    et = str(ldict.get("main_error_type", "")).lower()
    if et not in ALLOWED_ERROR_TYPE:
        raise ValueError(f"main_error_type invalid: {et}")
    return LabelPayload(
        main_language=ml,
        tatar_prob=_coerce_float(ldict.get("tatar_prob"), "tatar_prob"),
        russian_share=_coerce_float(ldict.get("russian_share"), "russian_share"),
        error_share=_coerce_float(ldict.get("error_share"), "error_share"),
        error_density=ed,
        main_error_type=et,
        non_fluent_prob=_coerce_float(ldict.get("non_fluent_prob"), "non_fluent_prob"),
        meaning_clarity=_coerce_float(ldict.get("meaning_clarity"), "meaning_clarity"),
        noise_score=_coerce_float(ldict.get("noise_score"), "noise_score"),
        overall_gec_usefulness=_coerce_float(ldict.get("overall_gec_usefulness"), "overall_gec_usefulness"),
    )


def _validate_items(data: Any) -> List[Dict[str, Any]]:
    """Validate a list response and return normalized label dicts."""
    if not isinstance(data, list):
        raise ValueError("Response must be a JSON list")
    validated: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each item must be an object")
        if "id" not in item or "labels" not in item:
            raise ValueError("Each item must contain 'id' and 'labels'")
        labels = _validate_labels(item["labels"])
        scored = ScoredItem(id=str(item["id"]), labels=labels)
        validated.append({"id": scored.id, "labels": asdict(scored.labels)})
    return validated


def call_gemini(
    client,
    model,
    prompt: str,
    schema,
) -> tuple[str, Optional[Any]]:
    """
    Call Gemini and return the raw text response and usage metadata. Caller should parse JSON.
    """
    config = types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        response_schema=schema,
        candidate_count=1,
        seed="1552",
        http_options={"timeout": 240 * 1000},
    )
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[prompt],
            config=config,
        )
        return resp.text, getattr(resp, "usage_metadata", None)
    except Exception as exc:  # pragma: no cover - network/runtime
        raise exc  # type: ignore


def parse_json_response(resp_text: str) -> Dict:
    """Parse and validate a Gemini JSON response, raising on mismatch."""
    text = resp_text.strip()
    if not text:
        raise ValueError("Empty response")
    if text.startswith("```"):
        text = text.strip("` \n")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        # Try bracketed list/object fallback; otherwise raise.
        start_list = text.find("[")
        end_list = text.rfind("]")
        if start_list != -1 and end_list != -1 and end_list > start_list:
            try:
                data = json.loads(text[start_list : end_list + 1])
            except Exception:
                raise ValueError(f"Failed to parse JSON list: {e}") from e
        else:
            start_obj = text.find("{")
            end_obj = text.rfind("}")
            if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
                try:
                    data = json.loads(text[start_obj : end_obj + 1])
                except Exception as e2:
                    raise ValueError(f"Failed to parse JSON object: {e2}") from e2
            else:
                raise ValueError(f"Failed to parse JSON: {e}") from e

    return _validate_items(data)


def build_batch_prompt(base_prompt: str, rows: Sequence[Dict[str, str]]) -> str:
    """Extend the base prompt with an input payload for batch scoring."""
    payload = {
        "task": "tatar_gec_scoring",
        "items": [
            {
                "id": str(row.get("id", "")),
                "text": str(row.get("text", "")),
            }
            for row in rows
        ],
    }
    return f"{base_prompt.strip()}\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
