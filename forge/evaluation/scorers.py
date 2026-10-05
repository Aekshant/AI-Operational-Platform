"""Quality scorers. Each returns a score in [0, 1] for (prediction, reference)."""

from __future__ import annotations

import re
import string
from collections import Counter

_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s: str) -> str:
    s = s.lower().translate(_PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, ref: str) -> float:
    return float(normalize(pred) == normalize(ref))


def contains(pred: str, ref: str) -> float:
    """Reference answer appears in the (possibly verbose) prediction: good for chat-style SLMs."""
    return float(normalize(ref) in normalize(pred))


def token_f1(pred: str, ref: str) -> float:
    p, r = normalize(pred).split(), normalize(ref).split()
    if not p or not r:
        return float(p == r)
    common = Counter(p) & Counter(r)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(r)
    return 2 * precision * recall / (precision + recall)


SCORERS = {"exact_match": exact_match, "contains": contains, "token_f1": token_f1}
