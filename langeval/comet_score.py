"""Semantic scoring with COMET (Unbabel/wmt22-comet-da).

chrF++ and BLEU reward surface overlap with the reference, so they penalise
valid paraphrases ("scenery is magnificent" vs "landscape is breathtaking").
COMET is a neural, reference-based metric that embeds source + hypothesis +
reference and scores MEANING equivalence — it credits correct-but-differently-
worded translations and correlates far better with human judgement.

It re-scores the generations already saved in results/raw_outputs/ — no model
re-runs. Runs locally on CPU (slow but fine for a few hundred sentences).

COMET-DA outputs ~0..1; we report it x100 (e.g. 0.84 -> 84.0) for visual parity
with chrF++ in the leaderboard. Reference-based COMET takes a single reference,
so we use the first Tatoeba reference per sentence.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_OUT = ROOT / "results" / "raw_outputs"
TESTSETS = ROOT / "data" / "testsets"

MODEL_NAME = "Unbabel/wmt22-comet-da"
_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        # COMET's predict() picks a "fork" dataloader context when MPS is
        # available, which crashes with num_workers=0 on Apple Silicon. We score
        # on CPU anyway (gpus=0), so neutralise the MPS check to get the standard
        # single-process loader.
        import torch
        torch.backends.mps.is_available = lambda: False
        from comet import download_model, load_from_checkpoint
        print(f"  loading COMET ({MODEL_NAME}) — first run downloads ~2.3 GB ...", flush=True)
        _MODEL = load_from_checkpoint(download_model(MODEL_NAME))
    return _MODEL


def _triples(model_id: str, lang: str):
    ts_path = TESTSETS / f"{lang}.json"
    raw_path = RAW_OUT / f"{model_id}__{lang}.jsonl"
    if not ts_path.exists() or not raw_path.exists():
        return []
    ts = {it["id"]: it for it in json.loads(ts_path.read_text(encoding="utf-8"))["items"]}
    data = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("hypothesis") and r["id"] in ts and ts[r["id"]]["refs"]:
            it = ts[r["id"]]
            data.append({"src": it["source"], "mt": r["hypothesis"], "ref": it["refs"][0]})
    return data


def comet_score(model_id: str, lang: str, batch_size: int = 16) -> dict | None:
    data = _triples(model_id, lang)
    if not data:
        return None
    out = _load_model().predict(data, batch_size=batch_size, gpus=0, progress_bar=False)
    return {"comet": round(out.system_score * 100, 2), "comet_n": len(data)}
