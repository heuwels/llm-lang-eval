"""Score a model's raw outputs against the multi-reference test set.

chrF++ (character n-gram F-score, word_order=2) is the headline metric — it is
the most robust simple metric for morphologically rich languages like Afrikaans
and German. BLEU is reported alongside for tradition/comparability. Both come
from sacreBLEU with recorded signatures for reproducibility.

Multi-reference: Tatoeba often links several valid English translations to one
source sentence. sacreBLEU needs a rectangular reference matrix, so we pad each
segment's reference list up to the max by repeating its last reference (a no-op
for scoring, since duplicate references don't change the score).
"""

import json
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

ROOT = Path(__file__).resolve().parent.parent
RAW_OUT = ROOT / "results" / "raw_outputs"
TESTSETS = ROOT / "data" / "testsets"


def _refs_by_id(lang: str) -> dict:
    data = json.loads((TESTSETS / f"{lang}.json").read_text(encoding="utf-8"))
    return {it["id"]: it["refs"] for it in data["items"]}


def score(model_id: str, lang: str) -> dict:
    refs_map = _refs_by_id(lang)
    path = RAW_OUT / f"{model_id}__{lang}.jsonl"

    hyps, ref_lists = [], []
    n_err = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("error") or rec.get("hypothesis") is None:
            n_err += 1
            continue
        rid = rec["id"]
        if rid in refs_map and refs_map[rid]:
            hyps.append(rec["hypothesis"])
            ref_lists.append(refs_map[rid])

    if not hyps:
        return {"model": model_id, "lang": lang, "n": 0, "errors": n_err,
                "note": "no scorable hypotheses"}

    max_refs = max(len(r) for r in ref_lists)
    ref_streams = [[r[i] if i < len(r) else r[-1] for r in ref_lists]
                   for i in range(max_refs)]

    chrf = CHRF(word_order=2)  # chrF++
    bleu = BLEU()
    chrf_s = chrf.corpus_score(hyps, ref_streams)
    bleu_s = bleu.corpus_score(hyps, ref_streams)

    return {
        "model": model_id, "lang": lang, "n": len(hyps), "errors": n_err,
        "chrf2": round(chrf_s.score, 2),
        "bleu": round(bleu_s.score, 2),
        "max_refs": max_refs,
        "chrf_sig": chrf.get_signature().format(short=True),
        "bleu_sig": bleu.get_signature().format(short=True),
    }
