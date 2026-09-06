#!/usr/bin/env python3
"""WER расшифровки и покрытие ошибок очередью проверки по прогонам скилла.

Для каждой записи выравнивает эталон с readable-сегментами, относит каждую
ошибку к окну и смотрит, попало ли окно в review_items. Печатает по режимам:
WER, долю окон и аудио на проверке, долю ошибок в отмеченных окнах, точность
отметок и покрытие записей."""
from __future__ import annotations
import json, re, sys, unicodedata
from pathlib import Path

# Использование: score.py EVAL_DIR GOLD.jsonl [PUBLIC_IDS.jsonl]
# EVAL_DIR/runs/<mode>/<id>/ — результаты скилла; GOLD — строки {"id", "gold"}.
EVAL = Path(sys.argv[1])
GOLD = Path(sys.argv[2])
PUBLIC = Path(sys.argv[3]) if len(sys.argv) > 3 else None

def normalize(text):
    text = unicodedata.normalize('NFKC', text).lower().replace('ё', 'е')
    text = re.sub(r'[^0-9a-zа-я]+', ' ', text, flags=re.IGNORECASE)
    return text.split()

def align(ref, hyp):
    """Возвращает список операций ('=', 'S', 'D', 'I') с индексами ref/hyp."""
    rows, cols = len(ref) + 1, len(hyp) + 1
    cost = [[0] * cols for _ in range(rows)]; op = [[''] * cols for _ in range(rows)]
    for i in range(1, rows): cost[i][0], op[i][0] = i, 'D'
    for j in range(1, cols): cost[0][j], op[0][j] = j, 'I'
    for i in range(1, rows):
        for j in range(1, cols):
            if ref[i-1] == hyp[j-1]:
                cost[i][j], op[i][j] = cost[i-1][j-1], '='; continue
            cost[i][j], op[i][j] = min(((cost[i-1][j-1]+1, 'S'), (cost[i-1][j]+1, 'D'), (cost[i][j-1]+1, 'I')), key=lambda t: t[0])
    ops = []; i, j = len(ref), len(hyp)
    while i or j:
        a = op[i][j]
        if a == '=': ops.append(('=', i-1, j-1)); i, j = i-1, j-1
        elif a == 'S': ops.append(('S', i-1, j-1)); i, j = i-1, j-1
        elif a == 'D': ops.append(('D', i-1, j)); i -= 1
        else: ops.append(('I', i, j-1)); j -= 1
    return ops[::-1]

gold = {json.loads(l)['id']: json.loads(l)['gold'] for l in GOLD.open()}
public = {json.loads(l)['id'] for l in PUBLIC.open()} if PUBLIC else set()
suite_of = lambda rid: 'public' if rid in public else 'personal'

def score_mode(mode):
    out = {s: dict(records=0, ref=0, errors=0, errors_flagged=0, segs=0, segs_flagged=0, segs_err=0, segs_err_flagged=0,
                   dur=0.0, dur_flagged=0.0, rec_err=0, rec_err_flagged=0, rec_clean=0, rec_clean_flagged=0, wall=0.0)
           for s in ('public', 'personal')}
    for rid, g in gold.items():
        d = EVAL / 'runs' / mode / rid
        if not (d/'segments.json').exists(): continue
        s = out[suite_of(rid)]; s['records'] += 1
        seg = json.loads((d/'segments.json').read_text())
        m = json.loads((d/'manifest.json').read_text())
        s['wall'] += m.get('timings_seconds', {}).get('total', 0)
        segments = seg['readable']; items = seg['review_items']
        ref = normalize(g)
        seg_words = [normalize(x['text']) for x in segments]
        hyp = [w for ws in seg_words for w in ws]
        owner = [k for k, ws in enumerate(seg_words) for _ in ws]
        ops = align(ref, hyp)
        seg_errors = [0] * len(segments)
        for a, i, j in ops:
            if a == '=': continue
            if a in ('S', 'I'): k = owner[j]
            else: k = owner[j] if j < len(owner) else (len(segments) - 1 if segments else 0)
            if segments: seg_errors[k] += 1
        flagged = [any(it['start'] < x['end'] and it['end'] > x['start'] for it in items) for x in segments]
        errs = sum(seg_errors)
        s['ref'] += len(ref); s['errors'] += errs
        s['errors_flagged'] += sum(e for e, f in zip(seg_errors, flagged) if f)
        s['segs'] += len(segments); s['segs_flagged'] += sum(flagged)
        s['segs_err'] += sum(1 for e in seg_errors if e); s['segs_err_flagged'] += sum(1 for e, f in zip(seg_errors, flagged) if e and f)
        for x, f in zip(segments, flagged):
            s['dur'] += x['end'] - x['start']; s['dur_flagged'] += (x['end'] - x['start']) if f else 0
        if errs: s['rec_err'] += 1; s['rec_err_flagged'] += bool(items)
        else: s['rec_clean'] += 1; s['rec_clean_flagged'] += bool(items)
    return out

for mode in sorted(d.name for d in (EVAL / 'runs').iterdir() if d.is_dir()):
    res = score_mode(mode)
    for suite, s in res.items():
        if not s['records']: continue
        pct = lambda a, b: f"{100*a/b:.1f}%" if b else '—'
        print(f"{mode:8} {suite:8} n={s['records']:2} WER={pct(s['errors'], s['ref'])} errors={s['errors']} "
              f"| flagged segs {s['segs_flagged']}/{s['segs']} ({pct(s['segs_flagged'], s['segs'])}), audio {pct(s['dur_flagged'], s['dur'])} "
              f"| errors in flagged {pct(s['errors_flagged'], s['errors'])} | segs-with-errors flagged {s['segs_err_flagged']}/{s['segs_err']} "
              f"| precision {pct(s['segs_err_flagged'], s['segs_flagged'])} | records w/err flagged {s['rec_err_flagged']}/{s['rec_err']}, clean flagged {s['rec_clean_flagged']}/{s['rec_clean']} "
              f"| wall {s['wall']:.0f}s")
