#!/usr/bin/env python3
"""Страница ручной разметки: по окну на карточку, обе гипотезы, поле с текстом GigaAM.

Использование: build_review.py EVAL_DIR, где EVAL_DIR/audio/<id>.wav и EVAL_DIR/runs/max/<id>/.
Экспорт даёт JSONL по окнам; склеить в эталон по записям: см. README."""
from __future__ import annotations
import html, json, sys
from pathlib import Path
from urllib.parse import quote

E = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
RUNS = E / 'runs/max'

def uri(p: Path) -> str: return 'file://' + quote(str(p.resolve()))

cards = []; total = 0
runs = [d for d in sorted(RUNS.iterdir()) if (d/'segments.json').exists()]
for d in runs:
    seg = json.loads((d/'segments.json').read_text()); man = json.loads((d/'manifest.json').read_text())
    audio = E/'audio'/f'{d.name}.wav'
    whisper = next((h for h in seg['hypotheses'] if 'whisper' in h['model']), None)
    wsegs = whisper['segments'] if whisper else []
    items = seg['review_items']
    dur = man['source']['duration_seconds']
    cards.append(f'<section id="{d.name}"><h2>{d.name} · {dur/60:.1f} мин · окон {len(seg["readable"])}</h2>'
                 f'<audio id="a-{d.name}" controls preload="metadata" src="{uri(audio)}"></audio>')
    for k, s in enumerate(seg['readable']):
        total += 1
        w = next((x for x in wsegs if x['start'] < s['end'] and x['end'] > s['start']), None)
        flagged = any(it['start'] < s['end'] and it['end'] > s['start'] for it in items)
        wid = f'{d.name}#{k}'
        tag = '· <span class="tag">в очереди проверки</span>' if flagged else ''
        cards.append(f'''<article class="card{' flagged' if flagged else ''}" data-id="{html.escape(wid)}" data-start="{s['start']}" data-end="{s['end']}" data-audio="a-{d.name}">
          <h3>{s['start']:.1f}–{s['end']:.1f} с {tag}<button class="play" type="button">▶ окно</button></h3>
          <div class="hyp"><b>GigaAM (readable)</b><span>{html.escape(s['text'])}</span></div>
          <div class="hyp"><b>Whisper</b><span>{html.escape(w['text'] if w else '—')}</span></div>
          <label>Эталон (поправь ошибки; филлеры «э-э», «хмм» не пиши)<textarea class="gold" rows="2">{html.escape(s['text'])}</textarea></label>
        </article>''')
    cards.append('</section>')
nav = ''.join(f'<a href="#{d.name}">{d.name[-6:]}</a> ' for d in runs)
body = ''.join(cards)
page = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Разметка записей</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Ctext y='26' font-size='26'%3E%F0%9F%8E%99%3C/text%3E%3C/svg%3E">
<style>
:root{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#18212f;background:#f7f8fa}}
body{{max-width:1100px;margin:auto;padding:24px}}h1{{margin:0 0 6px}}h2{{margin:28px 0 8px}}
nav{{position:sticky;top:0;background:#f7f8fadd;padding:10px 0;backdrop-filter:blur(8px);z-index:2}}
nav a,button{{border:0;border-radius:8px;padding:8px 12px;background:#2f54c9;color:#fff;text-decoration:none;cursor:pointer;margin-right:6px;font:inherit}}
button.play{{background:#e9edf7;color:#2f54c9;margin-left:10px;padding:4px 10px;font-size:13px}}
.card{{background:#fff;border:1px solid #dfe3ea;border-radius:12px;padding:14px 16px;margin:10px 0}}.card.flagged{{border-color:#e0a800;background:#fffdf3}}
.card h3{{margin:0 0 8px;font-size:15px}}.tag{{color:#8a6d00;font-weight:500;font-size:13px}}
.hyp{{display:grid;grid-template-columns:150px 1fr;gap:10px;padding:5px 0;border-top:1px solid #eef1f5}}.hyp b{{font-weight:600;color:#475467}}
label{{display:block;margin-top:8px;font-size:13px;color:#475467}}textarea{{box-sizing:border-box;width:100%;margin-top:4px;padding:8px;border:1px solid #bac3d2;border-radius:8px;font:inherit;font-size:15px}}
textarea.edited{{border-color:#2f54c9}}audio{{width:100%}}.muted{{color:#667085}}#status{{margin-left:8px;color:#475467}}
</style></head><body>
<h1>Разметка записей · {total} окон</h1>
<p class="muted">Кнопка «▶ окно» играет только это окно. В поле уже стоит текст GigaAM: поправь ошибки, лишнее удали, пропущенное добавь. Жёлтые карточки — окна, которые скилл сам поставил в очередь проверки. Правки сохраняются в браузере; «Экспорт» скачивает JSONL для пересчёта.</p>
<nav>{nav}<button id="export" type="button">Экспорт JSONL</button><span id="status"></span></nav>
{body}
<script>
const key='review-v1';const saved=JSON.parse(localStorage.getItem(key)||'{{}}');
document.querySelectorAll('.card').forEach(card=>{{const id=card.dataset.id;const ta=card.querySelector('.gold');const initial=ta.value;
  if(saved[id]!==undefined){{ta.value=saved[id];ta.classList.toggle('edited',saved[id]!==initial);}}
  ta.addEventListener('input',()=>{{saved[id]=ta.value;ta.classList.toggle('edited',ta.value!==initial);localStorage.setItem(key,JSON.stringify(saved));}});
  card.querySelector('.play').addEventListener('click',()=>{{const a=document.getElementById(card.dataset.audio);const end=+card.dataset.end;a.currentTime=+card.dataset.start;a.play();
    const stop=()=>{{if(a.currentTime>=end){{a.pause();a.removeEventListener('timeupdate',stop);}}}};a.addEventListener('timeupdate',stop);}});
}});
document.querySelector('#export').addEventListener('click',()=>{{
  const rows=[];document.querySelectorAll('.card').forEach(card=>{{rows.push(JSON.stringify({{id:card.dataset.id,start:+card.dataset.start,end:+card.dataset.end,gold:card.querySelector('.gold').value}}));}});
  const url=URL.createObjectURL(new Blob([rows.join('\\n')+'\\n'],{{type:'application/x-ndjson;charset=utf-8'}}));
  const a=document.createElement('a');a.href=url;a.download='adjudication.jsonl';document.body.appendChild(a);a.click();
  document.querySelector('#status').textContent='Экспортировано окон: '+rows.length;setTimeout(()=>{{a.remove();URL.revokeObjectURL(url);}},1000);
}});
</script></body></html>'''
(E/'review.html').write_text(page, encoding='utf-8')
print('окон', total, '→', E/'review.html')
