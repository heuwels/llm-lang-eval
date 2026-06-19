"""Parroting probe: blank an informative word, ask the model to fill it, measure
exact recovery — on seen (pre-cutoff) vs unseen (post-cutoff) sentences.

On unseen sentences a model can only predict the blank from context. If it
recovers the *exact* original word far more often on seen sentences, that gap is
memorisation — "parroting from params" rather than reasoning about the language.
recovery(pre) − recovery(post) per model is the signal. (It's also a direct
cloze-ability test — Lector's practice feature.)

Reuses the standard runner: the fill-blank answer rides in the same
{"translation": ...} JSON field, so no new schema/parse path is needed.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTSETS = ROOT / "data" / "testsets"
RAW_OUT = ROOT / "results" / "raw_outputs"
CLOZE_SCORES = ROOT / "results" / "cloze.json"

# short function-word stoplists so we mask a content word, not "the"/"and"
STOP = set((
    "die en van het is in op te dat met nie vir sy om as ek jy ons hulle was "  # afr
    "der die das und ist in zu den dem ein eine nicht mit auf für sich auch "    # deu
    "el la de que y en los las un una no se con por para su lo es al "           # spa
).split())

WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]+(?:'[A-Za-z]+)?")


def _pick_word(text: str) -> str | None:
    """Pick the longest content word (>=5 chars, not a stopword) to blank —
    longer/rarer words are least predictable, so they best expose memorisation."""
    words = WORD_RE.findall(text)
    cands = [w for w in words if len(w) >= 5 and w.lower() not in STOP]
    if not cands:
        cands = [w for w in words if len(w) >= 4 and w.lower() not in STOP]
    return max(cands, key=len) if cands else None


def build_masked(stem: str) -> Path:
    """Read data/testsets/<stem>.json, blank one word per item, write
    <stem>-cloze.json with source=masked sentence, answer=the blanked word."""
    data = json.loads((TESTSETS / f"{stem}.json").read_text(encoding="utf-8"))
    items = []
    for it in data["items"]:
        w = _pick_word(it["source"])
        if not w:
            continue
        masked = re.sub(rf"\b{re.escape(w)}\b", "___", it["source"], count=1)
        if "___" not in masked:
            continue
        items.append({"id": it["id"], "source": masked, "answer": w})
    out = TESTSETS / f"{stem}-cloze.json"
    out.write_text(json.dumps(
        {"meta": {"lang": data["meta"]["lang"],
                  "language_name": data["meta"]["language_name"], "task": "cloze",
                  "from": stem, "n": len(items)}, "items": items},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def cloze_prompt(masked: str, language_name: str) -> str:
    return (f"This {language_name} sentence has exactly one word replaced by ___. "
            f"Reply with ONLY the single {language_name} word that fills the blank.\n\n{masked}")


# A dedicated schema — the field is "word", not "translation", or models translate
# the masked sentence instead of filling the blank.
CLOZE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "cloze", "strict": True,
        "schema": {"type": "object", "properties": {"word": {"type": "string"}},
                   "required": ["word"], "additionalProperties": False},
    },
}


def parse_word(raw: str) -> str:
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("word"), str):
            return obj["word"].strip()
    except (json.JSONDecodeError, TypeError):
        pass
    toks = (raw or "").strip().split()
    return toks[0].strip(" .,\"'“”") if toks else ""


def _norm(s: str) -> str:
    return re.sub(r"[^\w]", "", (s or "").split()[0] if (s or "").split() else "").lower()


def recovery(model_id: str, stem: str) -> dict | None:
    """Exact-recovery rate of the blanked word from a cloze run."""
    ts_path = TESTSETS / f"{stem}-cloze.json"
    raw_path = RAW_OUT / f"{model_id}__{stem}-cloze.jsonl"
    if not ts_path.exists() or not raw_path.exists():
        return None
    gold = {it["id"]: it["answer"] for it in json.loads(ts_path.read_text())["items"]}
    n = hit = 0
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("error") or not r.get("hypothesis") or r["id"] not in gold:
            continue
        n += 1
        if _norm(r["hypothesis"]) == _norm(gold[r["id"]]):
            hit += 1
    return {"recovery": round(100 * hit / n, 1) if n else None, "n": n}
