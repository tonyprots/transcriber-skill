from __future__ import annotations

import re
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from .fillers import is_filler_token, strip_fillers
from .models import Hypothesis, ReviewItem, Segment


_PUNCTUATION = re.compile(r"[^\w\s-]+", re.UNICODE)

# Слова-паразиты. Их расхождение — не работа для человека: одна модель
# услышала «uh», другая промолчала, а читаемая версия их всё равно убирает.
# На часовом интервью это 378 расхождений из 1068 (замер 2026-09-20).
# Звуки вроде «хмм», «ммм», «эмм», «uhh» узнаёт `fillers.is_filler_token` —
# то же правило, по которому их убирает readable; здесь — слова и обороты,
# которые readable оставляет, но спорить о которых не о чем.
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

# Число, а не омофон: «to» и «for» в `_NUMBER_SCALE` ради «P to P», но рядом
# с ними числа нет.
_NUMBER_WORDS = frozenset(_NUMBER_SCALE) - {"to", "too", "for"}

# Единицы измерения при числе: цифровая запись теряет «%» и «$» ещё при
# нормализации, словесная оставляет «percent» и «dollars». Спорить тут не о
# чем — если рядом число и единица названа одна: «рублей» против «долларов»
# уже разные деньги.
_MEASURE_WORDS = {
    "percent": "%", "percents": "%", "percentage": "%",
    "процент": "%", "процента": "%", "процентов": "%",
    "dollars": "$", "dollar": "$", "долларов": "$", "рублей": "₽",
}

# Порядок числа в любом падеже: «миллиона», «миллиардов», «тысячи».
_SCALE_STEMS = {
    "тысяч": "k", "thousand": "k", "миллион": "m", "million": "m",
    "миллиард": "b", "billion": "b", "триллион": "t", "trillion": "t",
}
_SCALE_STEM = re.compile("^(" + "|".join(_SCALE_STEMS) + ")")

# Сколько слов контекста показывать по краям расхождения. Четыре — столько,
# чтобы понять фразу, и не столько, чтобы снова читать всё окно.
_CONTEXT_WORDS = 4

# Выше этого лексического сходства расхождение считается разным написанием
# одного слова («Airtable» и «Air Table»), а не разными словами.
_SPELLING_SIMILARITY = 0.62

# Виды расхождений, которые не считаются работой для человека: в отчёте они
# сворачиваются в одну строку, в `segments.json` остаются целиком.
MINOR_KINDS = frozenset({"filler", "equivalent", "disfluency", "verifier_artifact"})

# Фразы, которые Whisper дописывает в тишину или на краю окна: следы
# субтитров из его обучающих данных. В наших прогонах «Продолжение
# следует...» встретилось 19 раз, подписи редакторов субтитров — ещё 9; ни
# одна не была сказана. Признак срабатывает, только если у основной модели
# на этом месте пусто, а фраза занимает почти весь фрагмент.
_VERIFIER_ARTIFACTS = re.compile(
    r"продолжение следует|dimatorzok|редактор субтитров|корректор а кулакова"
    r"|спасибо за просмотр|подписывайтесь на (наш )?канал|ставьте лайки"
)
_ARTIFACT_SPARE_WORDS = 2


def _verifier_artifact(left_key: str, right_key: str) -> bool:
    if left_key or not right_key:
        return False
    rest = _VERIFIER_ARTIFACTS.sub(" ", right_key)
    return rest != right_key and len(rest.split()) <= _ARTIFACT_SPARE_WORDS


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


_BOUNDARY_GAP_SECONDS = 1.0
_BOUNDARY_WORDS = 2


def _matched_words(left: str, right: str) -> int:
    left_keys = _word_tokens(left)[1]
    right_keys = _word_tokens(right)[1]
    return sum(
        block.size
        for block in SequenceMatcher(None, left_keys, right_keys).get_matching_blocks()
    )


def rebalance_boundaries(
    pairs: list[tuple[Segment, Segment | None]],
) -> list[tuple[Segment, Segment | None]]:
    """Возвращает слово проверяющей в то окно, где его услышала основная.

    Текст проверяющей раскладывается по окнам через пословные таймкоды
    (`packing.unpack_hypothesis`), а у Whisper они плывут на доли секунды.
    Слово на стыке окон уезжало к соседу, и одно и то же слово, услышанное
    обеими моделями, давало два пункта очереди: «выпало» в одном окне и
    «лишнее» в соседнем. Здесь на каждом стыке пробуется перенести одно-два
    крайних слова проверяющей через границу; перенос принимается, только если
    совпадений с основной на паре окон становится строго больше.

    Окно без слов проверяющей тоже принимает перенос: раскладка больше не
    копирует в него крайние слова соседей, и слово, которое Whisper сдвинул
    к соседу, возвращается сюда.
    """
    result = list(pairs)
    for index in range(len(result) - 1):
        (left_primary, left_verifier), (right_primary, right_verifier) = (
            result[index],
            result[index + 1],
        )
        donor = left_verifier or right_verifier
        if donor is None:
            continue
        if right_primary.start - left_primary.end > _BOUNDARY_GAP_SECONDS:
            continue
        left_words = left_verifier.text.split() if left_verifier else []
        right_words = right_verifier.text.split() if right_verifier else []

        def score(left: list[str], right: list[str]) -> int:
            return _matched_words(left_primary.text, " ".join(left)) + _matched_words(
                right_primary.text, " ".join(right)
            )

        # Перенос может опустошить окно целиком: на r7 чаще всего так и было —
        # в коротком окне с «М-м» у проверяющей стояло одно слово соседа.
        best = (score(left_words, right_words), left_words, right_words)
        for count in range(1, _BOUNDARY_WORDS + 1):
            candidates = []
            if count <= len(left_words):
                candidates.append((left_words[:-count], left_words[-count:] + right_words))
            if count <= len(right_words):
                candidates.append((left_words + right_words[:count], right_words[count:]))
            for left, right in candidates:
                current = score(left, right)
                if current > best[0]:
                    best = (current, left, right)
        _, left, right = best
        if left is not left_words:
            # Окно может остаться без слов проверяющей: всё, что Whisper в нём
            # «услышал», оказалось словом соседа («М-м.» против «Классный»).
            result[index] = (left_primary, _moved(left_verifier, donor, left_primary, left))
            result[index + 1] = (
                right_primary,
                _moved(right_verifier, donor, right_primary, right),
            )
    return result


def _moved(
    verifier: Segment | None, donor: Segment, primary: Segment, words: list[str]
) -> Segment | None:
    if not words:
        return None
    if verifier is None:
        # Пустое окно получает слова соседа: границы — свои, уверенность — его.
        return replace(donor, start=primary.start, end=primary.end, text=" ".join(words))
    return replace(verifier, text=" ".join(words))


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


def _is_number(word: str) -> bool:
    return word in _NUMBER_WORDS or any(char.isdigit() for char in word)


def _spoken_words(text: str) -> list[str]:
    """Слова, как они произнесены: дефис — способ записи, а не звук.

    «S-O-U-R-C-E» — слово по буквам, то же, что «S O U R C E»; «и-и-и» и
    «да-а» — протянутый звук, то же, что «и» и «да»; «60-70» — два числа.
    """
    words: list[str] = []
    for word in text.split():
        parts = [part for part in word.split("-") if part]
        if len(parts) > 1 and all(len(part) == 1 and part.isalpha() for part in parts):
            words.append(parts[0] if len(set(parts)) == 1 else "".join(parts))
        elif len(parts) > 1 and set(parts[1:]) == {parts[0][-1]}:
            words.append(parts[0])
        else:
            words.extend(parts)
    return words


def _equivalent(left: str, right: str, *, near_number: bool = False) -> bool:
    """Одно и то же сказанное, записанное по-разному, — не расхождение.

    `near_number` — в расхождении или вплотную к нему есть число. Только тогда
    отбрасываются единица измерения и отдельная буква — мусор трансдьюсера
    («n 95»). Без числа «А / Б», «я / ∅» и «в / к» — разные слова, а не «то же
    число записано иначе».
    """
    left_words, right_words = _spoken_words(left), _spoken_words(right)
    units = [
        {_MEASURE_WORDS[word] for word in words if word in _MEASURE_WORDS}
        for words in (left_words, right_words)
    ]
    if all(units) and units[0] != units[1]:
        return False

    def canonical(words: list[str]) -> str:
        if near_number:
            words = [
                word
                for word in words
                if word not in _MEASURE_WORDS and (len(word) > 1 or word.isdigit())
            ]
        joined = " ".join(_spell_numbers(words))
        # «800 000» — это разделитель тысяч (группа ровно из трёх цифр), а
        # «60 70» — диапазон: склеивать можно только первое.
        joined = re.sub(r"(?<=\d)\s+(?=\d{3}\b)", "", joined)
        # «8:00» — ровно восемь: нули минут не произносят.
        joined = re.sub(r"(?<=\d)\s+00\b", "", joined)
        return re.sub(r"\b([a-zа-я])\s+(?=[a-zа-я]\b)", r"\1", joined)

    return canonical(left_words) == canonical(right_words)


def _number_signature(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Числа фрагмента: группы цифр и порядки («миллиона» → m).

    Посимвольно «3,5%» и «35%», «миллиона» и «миллиарда» почти одинаковы, но
    это разные числа: такое место — слушать, а не «то же слово записано иначе».
    """
    scales = []
    for word in text.split():
        match = _SCALE_STEM.match(word)
        if match:
            scales.append(_SCALE_STEMS[match.group(1)])
    return tuple(re.findall(r"\d+", text)), tuple(scales)


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
    words = [
        word for word in result.split() if word not in FILLERS and not is_filler_token(word)
    ]
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
        if _verifier_artifact(left_key, right_key):
            kind = "verifier_artifact"
        elif _equivalent(
            left_key,
            right_key,
            near_number=any(
                _is_number(word)
                for key in left_keys[max(0, l_start - 1) : l_end + 1]
                + right_keys[max(0, r_start - 1) : r_end + 1]
                for word in key.split()
            ),
        ):
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
        elif (
            left_content
            and right_content
            and _number_signature(left_content) == _number_signature(right_content)
            and SequenceMatcher(None, left_content, right_content).ratio() >= _SPELLING_SIMILARITY
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
    for readable_segment, lexical_segment in rebalance_boundaries(
        align_segments(readable.segments, lexical.segments)
    ):
        if lexical_segment is None:
            if not without_fillers(normalize_text(strip_fillers(readable_segment.text)[0])):
                # У основной одни паразиты («М-м.»), у проверяющей пусто:
                # спорить не о чем, а пункт с весом окна занимал бы очередь.
                continue
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
                    primary_span=item.left,
                    verifier_span=item.right,
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
        return f"{label}: то же сказанное записано иначе"
    if item.kind == "verifier_artifact":
        return f"{label}: типовая выдумка проверяющей"
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
