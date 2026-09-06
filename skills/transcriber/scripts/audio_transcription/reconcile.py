from __future__ import annotations

import re
from dataclasses import replace
from difflib import SequenceMatcher

from .models import Hypothesis, ReviewItem, Segment


_PUNCTUATION = re.compile(r"[^\w\s-]+", re.UNICODE)


def normalize_text(text: str) -> str:
    text = text.casefold().replace("ё", "е")
    text = _PUNCTUATION.sub(" ", text)
    return " ".join(text.split())


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def align_segments(
    readable: list[Segment], lexical: list[Segment]
) -> list[tuple[Segment, Segment | None]]:
    groups: dict[int, list[Segment]] = {index: [] for index in range(len(readable))}
    for segment in lexical:
        overlaps = [
            max(0.0, min(current.end, segment.end) - max(current.start, segment.start))
            for current in readable
        ]
        if overlaps and max(overlaps) > 0:
            groups[overlaps.index(max(overlaps))].append(segment)
            continue
        if readable:
            midpoint = (segment.start + segment.end) / 2
            distances = [
                abs(midpoint - (current.start + current.end) / 2) for current in readable
            ]
            nearest = distances.index(min(distances))
            if distances[nearest] <= 0.5:
                groups[nearest].append(segment)

    pairs: list[tuple[Segment, Segment | None]] = []
    for index, current in enumerate(readable):
        candidates = groups[index]
        if not candidates:
            pairs.append((current, None))
            continue
        confidences = [
            segment.confidence for segment in candidates if segment.confidence is not None
        ]
        combined = Segment(
            start=min(segment.start for segment in candidates),
            end=max(segment.end for segment in candidates),
            text=" ".join(segment.text.strip() for segment in candidates),
            confidence=(sum(confidences) / len(confidences)) if confidences else None,
            speakers=tuple(
                sorted({speaker for segment in candidates for speaker in segment.speakers})
            ),
        )
        pairs.append((current, combined))
    return pairs


def differing_tokens(left: str, right: str) -> tuple[str, ...]:
    left_tokens = normalize_text(left).split()
    right_tokens = normalize_text(right).split()
    matcher = SequenceMatcher(None, left_tokens, right_tokens)
    result: list[str] = []
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        value = " / ".join(
            part for part in (
                " ".join(left_tokens[left_start:left_end]),
                " ".join(right_tokens[right_start:right_end]),
            ) if part
        )
        if value:
            result.append(value)
    return tuple(result)


def find_review_items(
    readable: Hypothesis,
    lexical: Hypothesis,
    *,
    threshold: float = 0.82,
    comparison_label: str = "Лексическая GigaAM",
) -> list[ReviewItem]:
    items: list[ReviewItem] = []
    for readable_segment, lexical_segment in align_segments(
        readable.segments, lexical.segments
    ):
        if lexical_segment is None:
            items.append(
                ReviewItem(
                    start=readable_segment.start,
                    end=readable_segment.end,
                    readable_text=readable_segment.text,
                    comparison_text="",
                    similarity=0.0,
                    differing_tokens=(normalize_text(readable_segment.text),),
                    reason=f"{comparison_label}: нет соответствующего сегмента",
                )
            )
            continue
        score = similarity(readable_segment.text, lexical_segment.text)
        token_differences = differing_tokens(
            readable_segment.text, lexical_segment.text
        )
        low_confidence = any(
            segment.confidence is not None and segment.confidence < 0.55
            for segment in (readable_segment, lexical_segment)
        )
        if token_differences or low_confidence:
            if score < threshold:
                reason = f"{comparison_label}: гипотезы сильно расходятся"
            elif token_differences:
                reason = f"{comparison_label}: различаются токены"
            else:
                reason = f"{comparison_label}: низкая уверенность модели"
            items.append(
                ReviewItem(
                    start=min(readable_segment.start, lexical_segment.start),
                    end=max(readable_segment.end, lexical_segment.end),
                    readable_text=readable_segment.text,
                    comparison_text=lexical_segment.text,
                    similarity=score,
                    differing_tokens=token_differences,
                    reason=reason,
                )
            )
    return items


def with_text(segment: Segment, text: str) -> Segment:
    return replace(segment, text=text)


def segments_for_windows(
    segments: list[Segment], windows: list[tuple[float, float]]
) -> list[Segment]:
    return [
        segment
        for segment in segments
        if any(min(segment.end, end) - max(segment.start, start) > 0 for start, end in windows)
    ]
