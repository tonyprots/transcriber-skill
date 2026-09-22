from __future__ import annotations

import re
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from .models import Hypothesis, ReviewItem, Segment


_PUNCTUATION = re.compile(r"[^\w\s-]+", re.UNICODE)

# Слова-паразиты. Их расхождение — не работа для человека: одна модель
# услышала «uh», другая промолчала, а читаемая версия их всё равно убирает.
# На часовом интервью это 378 расхождений из 1068 (замер 2026-09-20).
FILLERS = frozenset(
    {
        "uh", "um", "uhh", "umm", "ah", "er", "erm", "hmm", "mhm", "mm", "huh",
        "yeah", "yep", "okay", "ok", "right", "well", "like", "so", "you know",
        "i mean", "sort of", "kind of", "actually", "basically",
        "э", "эм", "ээ", "мм", "ну", "вот", "как бы", "типа", "значит",
        "то есть", "в общем", "короче", "да", "ага", "угу",
    }
)

# Служебные слова. Они не исключают место из очереди, но и не поднимают его
# наверх: спор о «it's it's there» — это заикание, спор о «labs» — содержание.
# Вес расхождения считается по знаменательным словам.
FUNCTION_WORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "as", "of", "at", "by", "for",
        "from", "in", "into", "on", "onto", "to", "with", "without", "that", "this",
        "these", "those", "there", "here", "it", "its", "it's", "i", "i'm", "i've",
        "you", "your", "we", "we're", "our", "they", "them", "their", "he", "she",
        "his", "her", "is", "are", "was", "were", "be", "been", "am", "do", "does",
        "did", "have", "has", "had", "will", "would", "can", "could", "should",
        "not", "no", "all", "just", "very", "then", "than", "s", "t", "re", "ve",
        "и", "а", "но", "или", "же", "бы", "ли", "не", "ни", "в", "во", "на", "с",
        "со", "к", "ко", "по", "за", "из", "от", "до", "для", "о", "об", "у", "при",
        "это", "этот", "эта", "те", "тот", "там", "тут", "он", "она", "они", "мы",
        "вы", "я", "его", "её", "их", "наш", "ваш", "быть", "был", "была", "были",
        "есть", "что", "как", "так", "уже", "ещё", "еще", "все", "всё",
    }
)

# Число словом и цифрой — одно и то же сказанное слово, записанное по-разному:
# Whisper пишет «Q4» и «P2P», трансдьюсеры — «Q four» и «P to P».
_NUMBER_SCALE = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "million": 10**6, "billion": 10**9,
    # Омофоны: «P to P» и «P2P» — одно и то же произнесённое.
    "to": 2, "too": 2, "for": 4,
    "ноль": 0, "один": 1, "два": 2, "три": 3, "четыре": 4, "пять": 5,
    "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10,
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
    "сто": 100, "тысяча": 1000, "тысяч": 1000, "миллион": 10**6,
    "миллионов": 10**6, "миллиард": 10**9,
}

# Единицы измерения при числе: цифровая запись теряет «%» и «$» ещё при
# нормализации, словесная оставляет «percent» и «dollars». Спорить тут не о чем.
_MEASURE_WORDS = frozenset(
    {
        "percent", "percents", "percentage", "dollars", "dollar",
        "процент", "процента", "процентов", "рублей", "долларов",
    }
)

# Сколько слов контекста показывать по краям расхождения. Четыре — столько,
# чтобы понять фразу, и не столько, чтобы снова читать всё окно.
_CONTEXT_WORDS = 4

# Выше этого лексического сходства расхождение считается разным написанием
# одного слова («Airtable» и «Air Table»), а не разными словами.
_SPELLING_SIMILARITY = 0.62

# Виды расхождений, которые не считаются работой для человека: в отчёте они
# сворачиваются в одну строку, в `segments.json` остаются целиком.
MINOR_KINDS = frozenset({"filler", "equivalent", "disfluency"})


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


@dataclass(frozen=True)
class Disagreement:
    """Одно разошедшееся место внутри окна, а не всё окно целиком.

    Сравнение идёт по нормализованным ключам, а наружу отдаётся исходное
    написание: из него словарь восстанавливает канон («PostgreSQL», не
    «postgresql»), и его же читает человек.
    """

    left: str
    right: str
    left_key: str
    right_key: str
    left_content: str
    right_content: str
    # Слова расхождения, которых нет в соседнем контексте: по ним считается
    # вес, потому что именно они несут новое.
    novel_left: str
    novel_right: str
    left_start: int
    left_end: int
    left_context: str
    right_context: str
    kind: str

    @property
    def weight(self) -> int:
        """Вес — по различным знаменательным словам, которых нет в контексте.

        Не по длине фрагмента: «it's uh it's it's there» длиннее, чем «labs»,
        а спорить там не о чем. Место со служебными словами остаётся в
        очереди с весом 1, но не занимает её верх.
        """
        return max(
            len(_content_words(self.novel_left)),
            len(_content_words(self.novel_right)),
            1,
        )

    @property
    def similarity(self) -> float:
        if not self.left_key or not self.right_key:
            return 0.0
        return SequenceMatcher(None, self.left_key, self.right_key).ratio()


def _spell_numbers(words: list[str]) -> list[str]:
    """«ninety five» → «95», «eight hundred thousand» → «800000».

    Одно и то же число: Whisper пишет цифрами, трансдьюсер — словами. Без
    этого верх очереди занимают «95%» против «ninety five percent», и
    настоящие расхождения тонут под ними.
    """
    result: list[str] = []
    current: int | None = None
    total = 0
    for word in words:
        value = _NUMBER_SCALE.get(word)
        if value is None:
            if current is not None or total:
                result.append(str(total + (current or 0)))
                current, total = None, 0
            result.append(word)
            continue
        if value == 100:
            current = (current or 1) * 100
        elif value >= 1000:
            total = (total + (current or 1)) * value
            current = None
        elif current is not None and value >= current:
            # «sixty seventy» — это диапазон «60–70», а не 130: складывать
            # можно только убывающее («ninety five»).
            result.append(str(total + current))
            current, total = value, 0
        else:
            current = (current or 0) + value
    if current is not None or total:
        result.append(str(total + (current or 0)))
    return result


def _equivalent(left: str, right: str) -> bool:
    """Одно и то же сказанное, записанное по-разному, — не расхождение."""

    def canonical(text: str) -> str:
        # Дефис — способ записи, а не звук: «S-O-U-R-C-E-R-Y» и «S O U R C E R Y»
        # произнесены одинаково. Отдельная буква — мусор трансдьюсера («n 95»).
        words = [
            word
            for word in text.replace("-", " ").split()
            if word not in _MEASURE_WORDS and (len(word) > 1 or word.isdigit())
        ]
        joined = " ".join(_spell_numbers(words))
        # «800 000» — это разделитель тысяч (группа ровно из трёх цифр), а
        # «60 70» — диапазон: склеивать можно только первое.
        joined = re.sub(r"(?<=\d)\s+(?=\d{3}\b)", "", joined)
        return re.sub(r"\b([a-zа-я])\s+(?=[a-zа-я]\b)", r"\1", joined)

    return canonical(left) == canonical(right)


def _content_words(text: str) -> set[str]:
    return {word for word in text.split() if word not in FUNCTION_WORDS}


def without_fillers(text: str) -> str:
    """Фрагмент без слов-паразитов — то, о чём модели спорят по существу.

    Считать вес расхождения по всем словам нельзя: «so we because we you know
    we we at» — это восемь слов и первое место в очереди, хотя спорить там не
    о чем. Сравниваются и ранжируются остатки без паразитов.
    """
    result = text
    for phrase in sorted((f for f in FILLERS if " " in f), key=len, reverse=True):
        result = re.sub(rf"\b{re.escape(phrase)}\b", " ", result)
    words = [word for word in result.split() if word not in FILLERS]
    return " ".join(words)


def disagreements(left: str, right: str) -> list[Disagreement]:
    """Разбирает пару гипотез на отдельные расхождения с контекстом.

    Раньше на этом месте стоял `differing_tokens`, и любое из этих расхождений
    отправляло на проверку окно целиком. На часовом интервью так набиралось
    254 окна из 272: слушать предлагалось всю запись.
    """
    left_words, left_keys = _word_tokens(left)
    right_words, right_keys = _word_tokens(right)
    found: list[Disagreement] = []
    for tag, l_start, l_end, r_start, r_end in SequenceMatcher(
        None, left_keys, right_keys
    ).get_opcodes():
        if tag == "equal":
            continue
        left_span = " ".join(left_words[l_start:l_end])
        right_span = " ".join(right_words[r_start:r_end])
        left_key = " ".join(left_keys[l_start:l_end])
        right_key = " ".join(right_keys[r_start:r_end])
        left_content = without_fillers(left_key)
        right_content = without_fillers(right_key)
        context = set(
            left_keys[max(0, l_start - _CONTEXT_WORDS) : l_start]
            + left_keys[l_end : l_end + _CONTEXT_WORDS]
            + right_keys[max(0, r_start - _CONTEXT_WORDS) : r_start]
            + right_keys[r_end : r_end + _CONTEXT_WORDS]
        )
        novel_left = [word for word in left_content.split() if word not in context]
        novel_right = [word for word in right_content.split() if word not in context]
        if _equivalent(left_key, right_key):
            kind = "equivalent"
        elif left_content == right_content:
            # Остатки совпали — разошлись только паразиты, даже если сам
            # фрагмент длинный.
            kind = "filler"
        elif not novel_left and not novel_right:
            # Все слова расхождения повторяют соседние: «it's uh it's it's
            # there» рядом со словом «there» — это заикание, а не спорное
            # место. Новой информации здесь нет ни с одной стороны.
            kind = "disfluency"
        elif left_content and right_content and (
            SequenceMatcher(None, left_content, right_content).ratio() >= _SPELLING_SIMILARITY
        ):
            kind = "spelling"
        else:
            kind = "substantive"
        found.append(
            Disagreement(
                left=left_span,
                right=right_span,
                left_key=left_key,
                right_key=right_key,
                left_content=left_content,
                right_content=right_content,
                novel_left=" ".join(novel_left),
                novel_right=" ".join(novel_right),
                left_start=l_start,
                left_end=l_end,
                left_context=_with_context(left_words, l_start, l_end, left_span),
                right_context=_with_context(left_words, l_start, l_end, right_span),
                kind=kind,
            )
        )
    return found


def _word_tokens(text: str) -> tuple[list[str], list[str]]:
    """Исходные слова и их нормализованные ключи, позиция в позицию."""
    words: list[str] = []
    keys: list[str] = []
    for word in text.split():
        key = normalize_text(word)
        if not key:
            continue
        words.append(word)
        keys.append(key)
    return words, keys


def _with_context(tokens: list[str], start: int, end: int, span: str) -> str:
    """Расхождение в окружении соседних слов — одинаковом с обеих сторон."""
    before = tokens[max(0, start - _CONTEXT_WORDS) : start]
    after = tokens[end : end + _CONTEXT_WORDS]
    middle = span or "∅"
    parts = [
        ("… " if start - _CONTEXT_WORDS > 0 else "") + " ".join(before),
        middle,
        " ".join(after) + (" …" if end + _CONTEXT_WORDS < len(tokens) else ""),
    ]
    return " ".join(part for part in parts if part.strip())


def _span_bounds(segment: Segment, item: Disagreement, token_count: int) -> tuple[float, float]:
    """Метки времени внутри окна — оценка по позиции слова, а не измерение.

    Пословных таймкодов у onnx-моделей нет, но окно на двадцать секунд с одним
    разошедшимся словом всё равно бесполезно: человеку нужно, куда мотать.
    """
    duration = max(segment.end - segment.start, 0.0)
    if token_count <= 0 or duration <= 0:
        return segment.start, segment.end
    start = segment.start + duration * (item.left_start / token_count)
    end = segment.start + duration * (max(item.left_end, item.left_start + 1) / token_count)
    return start, min(max(end, start + 0.3), segment.end)


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
                    weight=len(normalize_text(readable_segment.text).split()) or 1,
                    kind="window",
                )
            )
            continue
        score = similarity(readable_segment.text, lexical_segment.text)
        token_count = len(_word_tokens(readable_segment.text)[0])
        # Мелкие расхождения не выбрасываются, а помечаются: в отчёте они
        # сворачиваются в одну строку, но в `segments.json` остаются — скилл
        # не прячет расхождения, он их ранжирует.
        found = disagreements(readable_segment.text, lexical_segment.text)
        work = [item for item in found if item.kind not in MINOR_KINDS]
        low_confidence = any(
            segment.confidence is not None and segment.confidence < 0.55
            for segment in (readable_segment, lexical_segment)
        )
        for item in found:
            start, end = _span_bounds(readable_segment, item, token_count)
            items.append(
                ReviewItem(
                    start=start,
                    end=end,
                    readable_text=item.left_context,
                    comparison_text=item.right_context,
                    similarity=item.similarity,
                    differing_tokens=(
                        " / ".join(part for part in (item.left, item.right) if part),
                    ),
                    reason=_reason(comparison_label, item, score, threshold),
                    weight=item.weight,
                    kind=item.kind,
                )
            )
        if not work and low_confidence:
            items.append(
                ReviewItem(
                    start=min(readable_segment.start, lexical_segment.start),
                    end=max(readable_segment.end, lexical_segment.end),
                    readable_text=readable_segment.text,
                    comparison_text=lexical_segment.text,
                    similarity=score,
                    differing_tokens=(),
                    reason=f"{comparison_label}: низкая уверенность модели",
                    weight=1,
                    kind="window",
                )
            )
    return items


def _reason(label: str, item: Disagreement, window_score: float, threshold: float) -> str:
    if item.kind == "filler":
        return f"{label}: расходятся слова-паразиты"
    if item.kind == "disfluency":
        return f"{label}: повтор или обрыв речи"
    if item.kind == "equivalent":
        return f"{label}: то же число записано иначе"
    if not item.right:
        return f"{label}: слова нет в проверочной гипотезе"
    if not item.left:
        return f"{label}: проверочная гипотеза слышит лишнее слово"
    if item.kind == "spelling":
        return f"{label}: то же слово записано иначе"
    if window_score < threshold:
        return f"{label}: гипотезы сильно расходятся, окно целиком под вопросом"
    words = item.weight
    return f"{label}: расходятся {words} {_plural(words)}"


def _plural(count: int) -> str:
    if count % 10 == 1 and count % 100 != 11:
        return "слово"
    if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
        return "слова"
    return "слов"


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
