from __future__ import annotations

from collections import Counter
import math
import re


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def token_f1(reference: str, hypothesis: str) -> float:
    ref = Counter(tokenize(reference))
    hyp = Counter(tokenize(hypothesis))
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    overlap = sum((ref & hyp).values())
    precision = overlap / sum(hyp.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _lcs_length(left: list[str], right: list[str]) -> int:
    if len(right) > len(left):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, 1):
            current.append(
                previous[index - 1] + 1
                if token == other
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    length = _lcs_length(ref, hyp)
    precision = length / len(hyp)
    recall = length / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _ngrams(tokens: list[str], size: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1))


def sentence_bleu4(reference: str, hypothesis: str) -> float:
    """Deterministic smoothed BLEU-4 fallback; RadEval BLEU is preferred for publication."""
    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    precisions: list[float] = []
    for size in range(1, 5):
        ref_ngrams = _ngrams(ref, size)
        hyp_ngrams = _ngrams(hyp, size)
        clipped = sum((ref_ngrams & hyp_ngrams).values())
        total = sum(hyp_ngrams.values())
        precisions.append((clipped + 1.0) / (total + 1.0))
    brevity = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref) / len(hyp))
    return brevity * math.exp(sum(math.log(value) for value in precisions) / 4)


def per_case_basic(reference: str, hypothesis: str) -> dict[str, float]:
    return {
        "token_f1": token_f1(reference, hypothesis),
        "rouge_l_f1_fallback": rouge_l_f1(reference, hypothesis),
        "bleu4_fallback": sentence_bleu4(reference, hypothesis),
        "exact_match": float(reference.strip().lower() == hypothesis.strip().lower()),
    }

