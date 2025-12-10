from typing import Iterable, List, Sequence, Tuple

from transformers import pipeline


def _is_toxic_label(label: str) -> bool:
    """
    Detect toxicity label across common naming schemes.
    """
    l = label.lower()
    return (
        "toxic" in l
        or l in {"label_1", "1", "offensive", "abusive", "hate", "insult"}
        or l.endswith("1")
    )


def load_toxicity_pipeline(device: int = -1):
    """
    Load toxicity classifier from textdetox repository.
    """
    return pipeline(
        task="text-classification",
        model="textdetox/bert-multilingual-toxicity-classifier",
        device=device,
    )


def extract_label_score(result) -> Tuple[int, float]:
    """
    Extract binary label (0=non-toxic, 1=toxic) and score from HF pipeline result.
    """
    # print(result)
    # if isinstance(result, list):
    #     if result and isinstance(result[0], dict) and "label" in result[0]:
    #         best = max(result, key=lambda x: x["score"])
    #         print(best)
    #         exit()
    #         return (1 if _is_toxic_label(best.get("label", "")) else 0, float(best.get("score", 0.0)))
    #     # Unexpected shape: mark non-toxic.
    #     return 0, 0.0

    label = result.get("label", "")
    if label == 'LABEL_1':
        is_toxic=1
    elif label == "LABEL_0":
        is_toxic=0
    else: raise ValueError(f"Unexpected label value '{label}'")
    score = float(result.get("score", 0.0))
    return (is_toxic, score)


def predict_toxicity(classifier, texts: Sequence[str]) -> List[Tuple[int, float]]:
    outputs = classifier(list(texts), truncation=True)
    return [extract_label_score(out) for out in outputs]
