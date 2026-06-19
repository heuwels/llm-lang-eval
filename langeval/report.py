"""Generate a self-contained, blog-post-style HTML eval report.

Mirrors the shape of a code/reasoning-model benchmark writeup:
  - headline leaderboard (chrF++ / BLEU per model x language)
  - per-language bar charts (inline SVG, no JS/CDN)
  - side-by-side model GENERATIONS for sentences where models disagree most
  - methodology + caveats + a one-line recommendation

Reads results/scores.json, results/raw_outputs/*.jsonl, and data/testsets/*.json.
Output is one portable HTML file (inline CSS, light/dark via prefers-color-scheme)
that can be published as-is or dropped into the Lector site's /blog.
"""

import html
import json
import os
import re
from pathlib import Path

import yaml
from sacrebleu.metrics import CHRF

ROOT = Path(__file__).resolve().parent.parent
SCORES = Path(os.environ["LANGEVAL_SCORES"]) if os.environ.get("LANGEVAL_SCORES") \
    else ROOT / "results" / "scores.json"
RAW = ROOT / "results" / "raw_outputs"
TESTSETS = ROOT / "data" / "testsets"
CONFIG = ROOT / "config" / "models.yaml"
CLOZE = ROOT / "results" / "cloze.json"
OUT = ROOT / "results" / "report.html"

LANG_NAMES = {"afr": "Afrikaans", "deu": "German", "spa": "Spanish"}
_CHRF = CHRF(word_order=2)
# COMET differences smaller than this are within sampling noise at n=200 (afr
# largely single-reference): treat the top as a band/tie, not a strict ranking.
NOISE = 1.5

# deployment tier per config section, for colour-coding
_TIER_OF_SECTION = {"local": "box", "ondevice": "ondevice", "cloud": "cloud",
                    "cloud_reference": "cloud"}
_TIER_LABEL = {"ondevice": "on-device (laptop)", "box": "self-hosted box (18 GB)",
               "cloud": "cloud (OpenRouter)"}
# dot-plot: one colour per language (distinct from the tier palette)
LANG_DOT = {"afr": "#e11d48", "deu": "#64748b", "spa": "#16a34a"}
TIER_FILL = {"ondevice": "#8b5cf6", "box": "#0ea5e9", "cloud": "#f59e0b"}


def _tiers() -> dict:
    if not CONFIG.exists():
        return {}
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    out = {}
    for sec, tier in _TIER_OF_SECTION.items():
        for m in (cfg.get(sec) or []):
            out[m["id"]] = tier
    return out


def _esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _load_scores() -> list:
    return json.loads(SCORES.read_text(encoding="utf-8")) if SCORES.exists() else []


def _load_hyps(model: str, lang: str) -> dict:
    f = RAW / f"{model}__{lang}.jsonl"
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("hypothesis"):
            out[r["id"]] = r["hypothesis"]
    return out


def _load_testset(lang: str) -> dict:
    f = TESTSETS / f"{lang}.json"
    if not f.exists():
        return {}
    data = json.loads(f.read_text(encoding="utf-8"))
    return {it["id"]: it for it in data["items"]}


def _sent_chrf(hyp: str, refs: list) -> float:
    from .score import normalize
    return _CHRF.sentence_score(normalize(hyp), [normalize(r) for r in refs]).score


def _ok_translation(h: str) -> bool:
    """Reject non-translation output (preamble/refusal/meta leakage) so the
    side-by-side compares actual translations, not instruction-following misses."""
    h = (h or "").strip()
    if len(h) < 2 or h.endswith(":"):
        return False
    return not re.match(
        r"(?i)^(in english|translation|english|here (is|are)|the (afrikaans|german|spanish)\b|"
        r"i (can|cannot|can't|'m sorry|am unable)|sorry|note:|sure[,!])", h)


# ---- HTML building blocks --------------------------------------------------

def _bar_chart(rows: list, value_key: str = "chrf2", max_val: float = 100.0) -> str:
    """rows: [{'label','chrf2','tier'}]. Horizontal SVG bars, coloured by tier."""
    bar_h, gap, label_w, track_w, pad = 14, 4, 140, 340, 5
    w = label_w + track_w + 60
    h = pad * 2 + len(rows) * (bar_h + gap)
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" class="chart">']
    y = pad
    for r in rows:
        val = r.get(value_key) or 0
        bw = max(2, (val / max_val) * track_w)
        cls = f"bar tier-bar-{r['tier']}" if r.get("tier") else "bar"
        parts.append(
            f'<text x="{label_w - 8}" y="{y + bar_h * 0.7}" class="bar-label" '
            f'text-anchor="end">{_esc(r["label"])}</text>'
            f'<rect x="{label_w}" y="{y}" width="{track_w}" height="{bar_h}" class="bar-track"/>'
            f'<rect x="{label_w}" y="{y}" width="{bw:.1f}" height="{bar_h}" class="{cls}"/>'
            f'<text x="{label_w + bw + 6:.1f}" y="{y + bar_h * 0.7}" class="bar-val">{val:.1f}</text>'
        )
        y += bar_h + gap
    parts.append("</svg>")
    return "".join(parts)


def _leaderboard(scores: list, langs: list, models: list) -> str:
    by = {(s["model"], s["lang"]): s for s in scores}
    tiers = _tiers()
    has_comet = any("comet" in s for s in scores)
    primary = "comet" if has_comet else "chrf2"  # rank/highlight by the meaning metric when present
    best = {l: max((by[(m, l)].get(primary) for m in models if (m, l) in by and by[(m, l)].get(primary) is not None),
                   default=None) for l in langs}
    unit = "COMET · chrF++" if has_comet else "chrF++ · BLEU"
    head = "".join(f"<th>{_esc(LANG_NAMES.get(l, l))}<br><span class='unit'>{unit}</span></th>" for l in langs)
    rows = []
    for m in models:
        cells = []
        for l in langs:
            s = by.get((m, l))
            if not s:
                cells.append("<td class='na'>—</td>")
                continue
            pv = s.get(primary)
            # highlight the whole leading BAND (within NOISE of the best), not a single winner
            in_band = pv is not None and best[l] is not None and pv >= best[l] - NOISE
            hi = " td--best" if in_band else ""
            if has_comet:
                top = f"{s['comet']:.1f}" if s.get("comet") is not None else "—"
                sub = f"chrF {s['chrf2']:.1f}"
            else:
                top = f"{s['chrf2']:.1f}"
                sub = f"BLEU {s['bleu']:.1f}"
            cells.append(
                f"<td class='num{hi}'><span class='chrf'>{top}</span>"
                f"<span class='bleu'>{sub}</span></td>"
            )
        tier = tiers.get(m, "")
        dot = (f"<span class='tier tier-{tier}' title='{_TIER_LABEL.get(tier, '')}'></span>"
               if tier else "")
        rows.append(f"<tr><td class='model'>{dot}{_esc(m)}</td>{''.join(cells)}</tr>")
    return (f"<table class='board'><thead><tr><th>Model</th>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def _generations(lang: str, models: list, k: int = 5) -> str:
    ts = _load_testset(lang)
    hyps = {m: _load_hyps(m, lang) for m in models}
    present = [m for m in models if hyps[m]]
    if not present or not ts:
        return ""
    tiers = _tiers()
    from .score import normalize
    cands = []
    for sid, it in ts.items():
        groups: dict = {}
        for m in present:
            h = hyps[m].get(sid)
            if not h or not _ok_translation(h):  # skip non-translation output
                continue
            key = re.sub(r"[\s\W]+$", "", normalize(h).lower())  # fold quotes/punct/case
            g = groups.setdefault(key, {"text": h, "chrf": _sent_chrf(h, it["refs"]), "models": []})
            g["models"].append(m)
        # camps = wordings ≥2 models share; one-offs collapse to a tail line
        glist = sorted(groups.values(), key=lambda g: -len(g["models"]))
        camps = [g for g in glist if len(g["models"]) >= 2]
        if len(camps) < 2:  # need a real fork, not 24 one-off wordings (= noise)
            continue
        singles = [g for g in glist if len(g["models"]) == 1]
        cands.append({"source": it["source"], "refs": it["refs"],
                      "camps": camps, "singles": singles})
    # rank by the strength of the SECOND camp — a genuine alternative consensus
    cands.sort(key=lambda c: (-len(c["camps"][1]["models"]), -len(c["camps"])))
    cands = cands[:k]

    out = [f"<h3>{_esc(LANG_NAMES.get(lang, lang))} — where the models split</h3>",
           "<p class='muted'>Sentences where models split into clear <em>camps</em> — several agreeing on "
           "one wording, several on another (one-off wordings collapsed to a tail). Count × tier-dots per "
           "camp; green = within ~6 chrF of the closest-to-reference camp.</p>",
           "<div class='callout'><strong>Why they differ:</strong> almost none of this is error — it's "
           "paraphrase choice. A contraction vs the full form, 'by the end' vs 'before the end', one valid "
           "synonym over another — each camp diverges from the single crowd-sourced reference in its own "
           "way. The spread is widest on longer, structurally flexible sentences (more ways to order the "
           "English) and on the high-resource languages, which is exactly why the COMET (meaning) gaps are "
           "far smaller than the chrF (surface) gaps. Read the camps as equally-valid translations, not "
           "right-vs-wrong.</div>"]
    for c in cands:
        best = max(g["chrf"] for g in c["camps"])
        cards = []
        for g in c["camps"]:
            sc = g["chrf"]
            cls = "gen gen--best" if sc >= best - 6 else "gen"
            dots = "".join(f"<span class='tier tier-{tiers.get(m, '')}' title='{_esc(m)}'></span>"
                           for m in g["models"])
            names = ", ".join(_esc(m) for m in g["models"])
            cards.append(
                f"<div class='{cls}'><div class='gen-models'><span class='gen-count'>"
                f"{len(g['models'])}×</span>{dots}<span class='gen-score'>{sc:.0f}</span>"
                f"<div class='gen-names'>{names}</div></div>"
                f"<div class='gen-text'>{_esc(g['text'])}</div></div>"
            )
        if c["singles"]:
            smods = ", ".join(_esc(g["models"][0]) for g in c["singles"])
            cards.append(f"<div class='gen gen--tail'><span class='gen-count'>+{len(c['singles'])}</span> "
                         f"one-off wordings <span class='gen-names'>({smods})</span></div>")
        refs = " &nbsp;·&nbsp; ".join(_esc(x) for x in c["refs"])
        out.append(
            f"<div class='ex'><div class='ex-src'><span class='tag'>{_esc(lang)}</span>"
            f"{_esc(c['source'])}</div>"
            f"<div class='ex-ref'><span class='tag tag--ref'>ref</span>{refs}</div>"
            f"<div class='gens'>{''.join(cards)}</div></div>"
        )
    return "".join(out)


def _contamination(scores: list) -> str:
    """Per-model pre-2023 vs 2025-26 Afrikaans comparison (the holdout test)."""
    by = {(s["model"], s["lang"]): s for s in scores}
    models = {m for (m, l) in by if l == "afr-post" and (m, "afr-pre") in by}
    if not models:
        return ""
    has_comet = any(by[(m, "afr-post")].get("comet") is not None for m in models)
    metric, mname = ("comet", "COMET") if has_comet else ("chrf2", "chrF++")
    ordered = sorted(models, key=lambda m: -(by[(m, "afr-post")].get(metric) or 0))
    rows = []
    for m in ordered:
        pre, post = by[(m, "afr-pre")], by[(m, "afr-post")]
        pv, qv = pre.get(metric), post.get(metric)
        if pv is None or qv is None:
            continue
        delta = qv - pv
        cls = "delta-bad" if delta <= -3 else ("delta-warn" if delta <= -1 else "delta-ok")
        rows.append(f"<tr><td class='model'>{_esc(m)}</td>"
                    f"<td class='num'>{pv:.1f}</td><td class='num'>{qv:.1f}</td>"
                    f"<td class='num {cls}'>{delta:+.1f}</td></tr>")
    return (f"<table class='board'><thead><tr><th>Model</th>"
            f"<th>pre-2023<br><span class='unit'>{mname}</span></th>"
            f"<th>2025–26<br><span class='unit'>{mname}</span></th>"
            f"<th>Δ</th></tr></thead><tbody>{''.join(rows)}</tbody></table>")


def _cloze_panel() -> str:
    """Per-model exact word-recovery on seen (afr-pre) vs unseen (afr-post)."""
    if not CLOZE.exists():
        return ""
    by = {(x["model"], x["set"]): x for x in json.loads(CLOZE.read_text(encoding="utf-8"))}
    models = [m for (m, s) in by if s == "afr-post" and (m, "afr-pre") in by]
    tiers = _tiers()
    rows = []
    for m in sorted(set(models)):
        pre, post = by[(m, "afr-pre")].get("recovery"), by[(m, "afr-post")].get("recovery")
        if pre is None or post is None:
            continue
        rows.append((m, pre, post, pre - post))
    if not rows:
        return ""
    rows.sort(key=lambda r: -r[3])
    trs = []
    for m, pre, post, gap in rows:
        dot = (f"<span class='tier tier-{tiers.get(m, '')}'></span>" if tiers.get(m) else "")
        cls = "delta-bad" if gap >= 10 else ("delta-warn" if gap >= 5 else "delta-ok")
        trs.append(f"<tr><td class='model'>{dot}{_esc(m)}</td><td class='num'>{pre:.0f}</td>"
                   f"<td class='num'>{post:.0f}</td><td class='num {cls}'>{gap:+.0f}</td></tr>")
    return ("<table class='board'><thead><tr><th>Model</th>"
            "<th>seen<br><span class='unit'>recovery %</span></th>"
            "<th>unseen<br><span class='unit'>recovery %</span></th>"
            f"<th>gap</th></tr></thead><tbody>{''.join(trs)}</tbody></table>")


def _dot_plot(scores: list, langs: list) -> str:
    """Cleveland dot plot: one row per model (ranked), a dot per language on a
    shared ZOOMED axis so the bunched scores actually separate. Connector shows
    each model's cross-language spread; the name carries its tier dot."""
    by = {(s["model"], s["lang"]): s for s in scores}
    tiers = _tiers()
    metric = "comet" if any("comet" in s for s in scores) else "chrf2"
    focus = "afr" if "afr" in langs else langs[0]
    models = sorted({m for (m, l) in by if l == focus and by[(m, l)].get(metric) is not None},
                    key=lambda m: -by[(m, focus)][metric])
    vals = [by[(m, l)][metric] for m in models for l in langs
            if (m, l) in by and by[(m, l)].get(metric) is not None]
    if not vals:
        return ""
    xmin, xmax = int(min(vals)) - 1, int(max(vals)) + 1
    row_h, pad, label_w, plot_w, axis_h = 18, 10, 150, 380, 26
    W, H = label_w + plot_w + 30, pad * 2 + len(models) * row_h + axis_h

    def X(v):
        return label_w + (v - xmin) / (xmax - xmin) * plot_w

    p = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart dotplot">']
    for t in range(xmin, xmax + 1):
        if t % 2:
            continue
        x = X(t)
        p.append(f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{H - axis_h}" class="dp-grid"/>'
                 f'<text x="{x:.1f}" y="{H - axis_h + 13}" class="dp-axis" text-anchor="middle">{t}</text>')
    y = pad + row_h / 2
    for m in models:
        pts = [(l, by[(m, l)][metric]) for l in langs
               if (m, l) in by and by[(m, l)].get(metric) is not None]
        p.append(f'<text x="{label_w - 14}" y="{y + 3:.1f}" class="dp-name" text-anchor="end">{_esc(m)}</text>'
                 f'<circle cx="{label_w - 7}" cy="{y:.1f}" r="3.5" fill="{TIER_FILL.get(tiers.get(m, ""), "#999")}"/>')
        if pts:
            xs = [X(v) for _, v in pts]
            p.append(f'<line x1="{min(xs):.1f}" y1="{y:.1f}" x2="{max(xs):.1f}" y2="{y:.1f}" class="dp-conn"/>')
            for l, v in pts:
                p.append(f'<circle cx="{X(v):.1f}" cy="{y:.1f}" r="4.5" fill="{LANG_DOT.get(l, "#999")}">'
                         f'<title>{_esc(m)} · {_esc(LANG_NAMES.get(l, l))}: {v:.1f}</title></circle>')
        y += row_h
    p.append("</svg>")
    leg = " ".join(f'<span class="dp-leg"><span class="dp-lgd" style="background:{LANG_DOT[l]}"></span>'
                   f'{_esc(LANG_NAMES.get(l, l))}</span>' for l in langs)
    return (f'<div class="panel">{"".join(p)}<div class="dp-legend">{leg}'
            f'<span class="muted"> &nbsp;·&nbsp; {metric.upper()}, zoomed axis {xmin}–{xmax}; '
            f'dot before each name = tier</span></div></div>')


CSS = """
:root{--bg:#fafafa;--card:#fff;--text:#1a1a1a;--muted:#636363;--border:#e2e2e2;
--accent:#b45309;--accent-soft:#fde9d3;--best:#15803d;--best-soft:#dcfce7;--mono:#1e1e2e}
@media(prefers-color-scheme:dark){:root{--bg:#111;--card:#1a1a1a;--text:#e4e4e4;
--muted:#999;--border:#2a2a2a;--accent:#f59e0b;--accent-soft:#3a2a10;--best:#4ade80;--best-soft:#14301f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);line-height:1.65;
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:880px;margin:0 auto;padding:2.5rem 1.25rem 4rem}
h1{font-size:2rem;letter-spacing:-.02em;margin:0 0 .25rem}
h2{font-size:1.4rem;margin:2.75rem 0 1rem;padding-top:1rem;border-top:1px solid var(--border)}
h3{font-size:1.1rem;margin:1.75rem 0 .5rem}
.date{color:var(--muted);font-size:.9rem;margin-bottom:1.5rem}
.lead{font-size:1.1rem;color:var(--muted)}
a{color:var(--accent)}.muted{color:var(--muted);font-size:.92rem}
table.board{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.95rem}
.board th,.board td{border-bottom:1px solid var(--border);padding:.55rem .6rem;text-align:center}
.board th{font-size:.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.board th .unit{font-size:.72rem;letter-spacing:0;text-transform:none}
.board td.model{text-align:left;font-weight:600;font-family:var(--mono),monospace;font-size:.85rem}
.board td.num .chrf{font-weight:700}.board td.num .bleu{color:var(--muted);font-size:.8rem;margin-left:.4rem}
.board td.td--best{background:var(--best-soft)}.board td.td--best .chrf{color:var(--best)}
.board td.na{color:var(--muted)}
.chart{width:100%;height:auto}.bar-track{fill:var(--border);opacity:.5;rx:3}
.bar{fill:var(--accent);rx:3}.bar--best{fill:var(--best)}
.tier-bar-ondevice{fill:#8b5cf6}.tier-bar-box{fill:#0ea5e9}.tier-bar-cloud{fill:#f59e0b}
.bar-label{fill:var(--text);font-size:11px;font-family:var(--mono),monospace}
.bar-val{fill:var(--muted);font-size:11px}
.dotplot{width:100%;height:auto}
.dp-grid{stroke:var(--border);stroke-width:1;opacity:.5}
.dp-axis{fill:var(--muted);font-size:10px}
.dp-name{fill:var(--text);font-size:11px;font-family:var(--mono),monospace}
.dp-conn{stroke:var(--border);stroke-width:2}
.dp-legend{margin-top:.6rem;font-size:.82rem;color:var(--text)}
.dp-leg{margin-right:.9rem;white-space:nowrap}
.dp-lgd{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:.3rem;vertical-align:middle}
.panel{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:1rem 1.25rem;margin:1rem 0}
.ex{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:1rem;margin:.85rem 0}
.ex-src{font-size:1.02rem}.ex-ref{color:var(--muted);font-size:.92rem;margin-top:.2rem}
.tag{display:inline-block;background:var(--accent-soft);color:var(--accent);font-size:.66rem;font-weight:700;
text-transform:uppercase;letter-spacing:.05em;padding:.1rem .4rem;border-radius:4px;margin-right:.5rem;vertical-align:middle}
.tag--ref{background:transparent;color:var(--muted);border:1px solid var(--border)}
.gens{display:grid;gap:.5rem;margin-top:.75rem}
.gen{border:1px solid var(--border);border-radius:6px;padding:.5rem .7rem}
.gen--best{border-color:var(--best);background:var(--best-soft)}
.gen--tail{color:var(--muted);font-size:.85rem;border-style:dashed}
.gen-models{display:flex;align-items:center;flex-wrap:wrap;gap:.25rem;margin-bottom:.3rem}
.gen-count{font-weight:700;font-size:.8rem;margin-right:.2rem}
.gen-names{flex-basis:100%;font-family:var(--mono),monospace;font-size:.72rem;color:var(--muted);margin-top:.15rem}
.gen-score{margin-left:auto;background:var(--border);border-radius:999px;padding:0 .45rem;font-size:.72rem;font-weight:700}
.gen-text{font-size:.96rem}
.callout{background:var(--accent-soft);border-left:3px solid var(--accent);padding:.85rem 1.1rem;border-radius:0 6px 6px 0;margin:1rem 0;font-size:.95rem}
.footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--border);color:var(--muted);font-size:.85rem}
code{background:var(--border);padding:.1rem .35rem;border-radius:4px;font-size:.85em}
.delta-bad{color:#dc2626;font-weight:700}.delta-warn{color:#d97706;font-weight:700}
.delta-ok{color:var(--best);font-weight:700}
.tier{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;vertical-align:middle}
.tier-ondevice{background:#8b5cf6}.tier-box{background:#0ea5e9}.tier-cloud{background:#f59e0b}
.legend .tier{margin:0 .25rem 0 .5rem}
.warn{background:#fef2f2;border-left:3px solid #dc2626;padding:.85rem 1.1rem;border-radius:0 6px 6px 0;margin:1rem 0;font-size:.95rem}
@media(prefers-color-scheme:dark){.warn{background:#2a1416}}
"""


def generate(date: str = "", title: str = "") -> Path:
    scores = _load_scores()
    if not scores:
        raise SystemExit("no results/scores.json yet — run some models first")
    langs = [l for l in ("afr", "deu", "spa") if any(s["lang"] == l for s in scores)]
    # rank models by mean chrF++ across languages (desc)
    by_model: dict = {}
    for s in scores:
        by_model.setdefault(s["model"], []).append(s["chrf2"])
    models = sorted(by_model, key=lambda m: -sum(by_model[m]) / len(by_model[m]))

    focus = "afr" if "afr" in langs else langs[0]
    by = {(s["model"], s["lang"]): s for s in scores}
    n_focus = next((by[(m, focus)]["n"] for m in models if (m, focus) in by), 0)

    title = title or "Which local LLM translates best? A reproducible eval"
    date = date or "—"

    contam_table = _contamination(scores)
    contam_section = f"""
<h2>Contamination check: does it survive on unseen data?</h2>
<div class="warn"><strong>The honest limitation.</strong> Tatoeba is almost certainly in every model's
pretraining, so a high score can mean "translated well" <em>or</em> "regurgitated a memorised pair" — the
score alone can't tell us which. To bound it, each model is compared on two matched 150-sentence Afrikaans
samples (same length filter): <strong>pre-2023</strong> (added 2010–2022, almost certainly seen in training)
versus <strong>2025–26</strong> (added after the training cutoff of the older-generation models here, so they
cannot have memorised them). A large drop on the recent set is the fingerprint of memorisation; a stable
score is evidence of genuine translation ability.</div>
{contam_table}
<p class="muted">Caveat on the caveat: exact training-cutoff dates aren't published for every model, and
recently-added sentences may differ subtly in style or difficulty — so read a small Δ as "holds up", not as a
precise measurement of contamination.</p>
""" if contam_table else ""

    cloze_table = _cloze_panel()
    cloze_section = f"""
<h2>Parroting probe: memorisation, measured directly</h2>
<div class="warn"><strong>The sharpest contamination test.</strong> We blank one informative word per
sentence and ask each model to fill it. On <strong>unseen</strong> (2025–26) sentences it can only predict
from context; if it recovers the exact original word much more often on <strong>seen</strong> (pre-2023)
sentences, that gap is the model parroting memorised text rather than reasoning about the language. (It
doubles as a cloze-ability score — Lector's own practice task.)</div>
{cloze_table}
<p class="muted">Recovery = exact match of the blanked word. A large positive gap = memorisation; near-zero
= genuine context prediction. n≈150 per cell, so gaps within ~±10 are noise.</p>
""" if cloze_table else ""

    dotplot = _dot_plot(scores, langs)

    gens = "".join(_generations(l, models) for l in langs)

    # recommendation line — prefer the meaning metric (COMET) when present
    has_comet = any("comet" in s for s in scores)
    metric, mname = ("comet", "COMET") if has_comet else ("chrf2", "chrF++")
    top = sorted((s for s in scores if s["lang"] == focus and s.get(metric) is not None),
                 key=lambda s: -s[metric])
    rec = ""
    if top:
        topv = top[0][metric]
        band = [s for s in top if s[metric] >= topv - NOISE]
        tmap = _tiers()
        box_in_band = next((s for s in band if tmap.get(s["model"]) == "box"), None)
        rec = (f"On <strong>{_esc(LANG_NAMES.get(focus, focus))}</strong> the field is tightly "
               f"bunched: <strong>{len(band)} of {len(top)}</strong> models fall within ~{NOISE} "
               f"{mname} (sampling noise) of the top score (≈{topv:.0f}) — a statistical tie, "
               f"not a ranking.")
        if box_in_band:
            rec += (f" The self-hosted 18&nbsp;GB <code>{_esc(box_in_band['model'])}</code> "
                    f"({box_in_band[metric]:.1f}) sits in that band alongside frontier cloud — so "
                    f"for Afrikaans&rarr;English, you don't need the cloud or a big box.")

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title><style>{CSS}</style></head><body><div class="wrap">
<h1>{_esc(title)}</h1>
<p class="date">{_esc(date)} · source → English · Afrikaans / German / Spanish</p>
<p class="lead">A reproducible benchmark of on-device, self-hosted, and cloud LLMs on sentence
translation into English, built to choose a translation model for
<a href="https://github.com/heuwels/lector">Lector</a> — with the low-resource case (Afrikaans)
front and centre. Every model gets the same blinded Tatoeba source sentences, the same prompt,
greedy decoding, and is scored multi-reference with <strong>COMET</strong> (semantic) and
<strong>chrF++</strong> (surface), BLEU alongside.</p>

<div class="callout">{rec or "Run more models to populate the leaderboard."}</div>

<h2>Leaderboard</h2>
<p class="muted">All 24 models across three languages on one zoomed COMET axis — each row a model
(ranked by Afrikaans), one dot per language, the connector its cross-language spread. The zoom makes
the tight differences legible; gaps under ~{NOISE} COMET are sampling noise (see <em>Significance</em>).</p>
{dotplot}
<p class="muted legend">Deployment tier:<span class="tier tier-ondevice"></span>on-device (laptop)
<span class="tier tier-box"></span>self-hosted box (18 GB)
<span class="tier tier-cloud"></span>cloud (OpenRouter)</p>

<h3>The numbers</h3>
<p class="muted">COMET (meaning, ×100) over chrF++ (surface), per language. <strong>chrF++</strong>
rewards character overlap with the reference, so it docks valid paraphrases
(<em>"scenery is magnificent"</em> vs <em>"landscape is breathtaking"</em>); <strong>COMET</strong>
scores meaning and credits them. <strong>Green</strong> = leading band (within ~{NOISE} COMET — a
statistical tie, not a single winner). n={n_focus} per language.</p>
{_leaderboard(scores, langs, models)}

{contam_section}

{cloze_section}

<h2>Side-by-side generations</h2>
<p class="muted">The numbers only say so much. Here are the actual translations where models
disagree most — green = highest per-sentence chrF++ for that sentence.</p>
{gens}

<h2>Methodology</h2>
<div class="panel"><ul>
<li><strong>Task.</strong> Blinded source → English. The model sees only the source sentence; references are held out for scoring.</li>
<li><strong>Data.</strong> <a href="https://tatoeba.org">Tatoeba</a> sentence pairs (CC-BY 2.0 FR), seeded random sample, multi-reference where available, length-filtered.</li>
<li><strong>Prompt.</strong> One fixed user message, identical across models (no per-model tuning). Chain-of-thought disabled (<code>reasoning_effort: none</code>) — translation needs none.</li>
<li><strong>Structured output.</strong> Every model is constrained to emit <code>{{"translation": "…"}}</code> via a json_schema <code>response_format</code>. This is the equaliser: small models otherwise "think out loud" in plain text and bury the answer in preamble. Constrained decoding makes that impossible, gives every model the identical constraint, and mirrors how Lector itself prompts.</li>
<li><strong>Decoding.</strong> <code>temperature = 0</code> (greedy), one model resident at a time on an 18&nbsp;GB host (JIT load/evict).</li>
<li><strong>Metrics.</strong> chrF++ and BLEU via sacreBLEU (signatures recorded). COMET planned.</li>
</ul></div>

<h2>Caveats</h2>
<div class="panel"><ul>
<li><strong>Significance — read bands, not ranks.</strong> n={n_focus} per language (Afrikaans largely single-reference), so per-system COMET 95% confidence intervals are roughly ±1–2 points. Differences below ~{NOISE} COMET are sampling noise: the green leading band is a statistical tie and the sort order <em>within</em> it is not meaningful. Per-segment bootstrap CIs are future work.</li>
<li><strong>This is a proxy.</strong> It measures general sentence MT, not Lector's actual word/phrase dictionary-lookup task — a strong signal for model choice, not "Lector's output graded."</li>
<li><strong>Contamination — the big one.</strong> Tatoeba is in these models' pretraining, so a high score can reflect <em>memorising the pair</em> rather than reasoning about the language — and the score alone can't separate the two. The contamination-check section above bounds this with a post-cutoff holdout; treat absolute scores with suspicion and weight the pre-vs-post deltas and relative gaps over the headline numbers.</li>
<li><strong>Into-English is the easy direction</strong>, and Afrikaans here is largely single-reference. Read accordingly.</li>
</ul></div>

<div class="footer">Generated by <a href="https://github.com/heuwels/llm-lang-eval">llm-lang-eval</a>.
Harness + raw generations are open and reproducible.</div>
</div></body></html>"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc, encoding="utf-8")
    return OUT
