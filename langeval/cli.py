"""Command-line entry point.

  python -m langeval.cli fetch afr --n 200
  python -m langeval.cli run gemma-4-e4b --lang afr \
         --endpoint http://100.101.140.45:1234 --api-model google/gemma-4-e4b
  python -m langeval.cli score gemma-4-e4b --lang afr
  python -m langeval.cli report

`run` can also resolve --endpoint/--api-model from config/models.yaml by id
(--from-config), so a full sweep is scriptable.
"""

import argparse
import json
import os
from datetime import date
from pathlib import Path

import yaml

from . import prompts
from .fetch_tatoeba import LANG_NAMES, base_lang, build
from .runner import TranslateError, preflight, run
from .score import score

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "models.yaml"
# LANGEVAL_SCORES lets a parallel run write to an isolated scores file (merged later)
SCORES = Path(os.environ["LANGEVAL_SCORES"]) if os.environ.get("LANGEVAL_SCORES") \
    else ROOT / "results" / "scores.json"


def _load_env():
    """Load KEY=VALUE pairs from .env.local / .env into the environment so
    api keys (e.g. OPENROUTER_API_KEY) resolve. Existing env vars win."""
    for fn in (".env.local", ".env"):
        p = ROOT / fn
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}


# model entries can live in any of these sections
_SECTIONS = ("local", "ondevice", "cloud", "cloud_reference")


def _all_models(cfg: dict) -> list:
    out = []
    for sec in _SECTIONS:
        out += cfg.get(sec) or []  # section may be present-but-empty (None)
    return out


def _resolve(cfg: dict, m: dict) -> dict:
    """Resolve where/how to call a model: endpoint, api key, schema mode,
    concurrency (cloud APIs parallelise; a single local server stays serial)."""
    endpoint = m.get("endpoint") or cfg.get("defaults", {}).get("lmstudio_endpoint")
    return {
        "endpoint": endpoint,
        "api_key": os.environ.get(m["env_key"]) if m.get("env_key") else None,
        "no_schema": bool(m.get("no_schema")),
        "extra_body": m.get("extra_body"),
        "api_model": m.get("api_model", m["id"]),
        "concurrency": m.get("concurrency") or (8 if endpoint and "openrouter" in endpoint else 1),
    }


def cmd_fetch(a):
    build(a.lang, n=a.n, seed=a.seed, after=a.after, before=a.before, tag=a.tag)


def cmd_run(a):
    r = {"endpoint": a.endpoint, "api_model": a.api_model, "api_key": None,
         "no_schema": False, "extra_body": None}
    if a.from_config:
        cfg = _config()
        m = next((m for m in _all_models(cfg) if m["id"] == a.model_id), None)
        if m:
            res = _resolve(cfg, m)
            r = {k: (a.endpoint if k == "endpoint" and a.endpoint else
                     a.api_model if k == "api_model" and a.api_model else res[k]) for k in res}
    if not r["endpoint"] or not r["api_model"]:
        raise SystemExit("need --endpoint and --api-model (or --from-config with a match)")
    name = LANG_NAMES.get(base_lang(a.lang), a.lang)
    testset = ROOT / "data" / "testsets" / f"{a.lang}.json"
    if not testset.exists():
        raise SystemExit(f"no test set at {testset}; run `fetch {a.lang}` first")

    # Fail fast: one probe before firing the whole set.
    try:
        preflight(r["endpoint"], r["api_model"], name, extra_body=r["extra_body"],
                  no_schema=r["no_schema"], api_key=r["api_key"])
    except TranslateError as e:
        raise SystemExit(f"preflight failed for {a.model_id} @ {r['endpoint']}\n  {e}")

    run(a.model_id, r["endpoint"], r["api_model"], testset, name, prompts.build,
        limit=a.limit, extra_body=r["extra_body"], no_schema=r["no_schema"],
        api_key=r["api_key"], concurrency=r["concurrency"])


def _save_score(res: dict):
    data = json.loads(SCORES.read_text()) if SCORES.exists() else []
    data = [d for d in data if not (d["model"] == res["model"] and d["lang"] == res["lang"])]
    data.append(res)
    SCORES.parent.mkdir(parents=True, exist_ok=True)
    SCORES.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_score(a):
    res = score(a.model_id, a.lang)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    _save_score(res)


def cmd_sweep(a):
    """Run+score every on-host model (have: true) across the given languages,
    relying on LM Studio JIT auto-evict to load each in turn. Resilient: a model
    that fails preflight is skipped and logged, not allowed to abort the batch."""
    cfg = _config()
    if a.models:
        wanted = set(a.models)
        models = [m for m in _all_models(cfg) if m["id"] in wanted]
    else:
        models = [m for m in cfg.get("local", []) if m.get("have")]
    langs = a.langs or ["afr", "deu", "spa"]

    dropped = []
    for m in models:
        r = _resolve(cfg, m)
        for lang in langs:
            ts = ROOT / "data" / "testsets" / f"{lang}.json"
            if not ts.exists():
                print(f"SKIP {m['id']}/{lang}: no test set")
                continue
            name = LANG_NAMES.get(base_lang(lang), lang)
            try:
                preflight(r["endpoint"], r["api_model"], name, extra_body=r["extra_body"],
                          no_schema=r["no_schema"], api_key=r["api_key"])
            except TranslateError as e:
                print(f"DROP {m['id']}/{lang}: {e}")
                dropped.append({"model": m["id"], "lang": lang, "reason": str(e)[:160]})
                continue
            print(f"\n=== {m['id']} / {lang} ===")
            run(m["id"], r["endpoint"], r["api_model"], ts, name, prompts.build,
                limit=a.limit, extra_body=r["extra_body"], no_schema=r["no_schema"],
                api_key=r["api_key"], concurrency=r["concurrency"])
            _save_score(score(m["id"], lang))

    if dropped:
        print(f"\nDROPPED {len(dropped)} model/lang combos:")
        for d in dropped:
            print(f"  - {d['model']}/{d['lang']}: {d['reason']}")
    cmd_report(a)


def _print_text_table():
    if not SCORES.exists():
        print("no results/scores.json yet")
        return
    data = json.loads(SCORES.read_text())
    langs = sorted({d["lang"] for d in data})
    models = sorted({d["model"] for d in data}, key=lambda m: -sum(
        d["chrf2"] for d in data if d["model"] == m))
    print("\nchrF++ (higher is better)\n")
    header = f"{'model':<22}" + "".join(f"{l:>10}" for l in langs)
    print(header)
    print("-" * len(header))
    for m in models:
        row = f"{m:<22}"
        for l in langs:
            cell = next((d for d in data if d["model"] == m and d["lang"] == l), None)
            row += f"{cell['chrf2']:>10.1f}" if cell else f"{'-':>10}"
        print(row)


def cmd_comet(a):
    """Add COMET semantic scores to scores.json by re-scoring saved generations."""
    from .comet_score import comet_score
    if not SCORES.exists():
        raise SystemExit("no results/scores.json yet")
    data = json.loads(SCORES.read_text())
    targets = data if not a.models else [d for d in data if d["model"] in a.models]
    if getattr(a, "langs", None):
        targets = [d for d in targets if d["lang"] in a.langs]
    for d in targets:
        res = comet_score(d["model"], d["lang"])
        if res:
            d.update(res)
            print(f"  {d['model']:<18} {d['lang']}  COMET {res['comet']}  (n={res['comet_n']})")
    SCORES.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"updated {SCORES}")


def cmd_report(a):
    _print_text_table()
    if getattr(a, "text", False):
        return
    from .report import generate
    out = generate(date=getattr(a, "date", "") or date.today().isoformat(),
                   title=getattr(a, "title", "") or "")
    print(f"\nwrote {out}")


def main():
    _load_env()
    p = argparse.ArgumentParser(prog="langeval")
    sub = p.add_subparsers(required=True)

    f = sub.add_parser("fetch", help="build a blinded test set from Tatoeba")
    f.add_argument("lang", choices=list(LANG_NAMES))
    f.add_argument("--n", type=int, default=200)
    f.add_argument("--seed", type=int, default=13)
    f.add_argument("--after", help="keep only sentences added on/after YYYY-MM-DD")
    f.add_argument("--before", help="keep only sentences added before YYYY-MM-DD")
    f.add_argument("--tag", help="variant name -> data/testsets/<lang>-<tag>.json")
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("run", help="translate the test set with one model")
    r.add_argument("model_id")
    r.add_argument("--lang", required=True, help="test-set stem, e.g. afr or afr-post")
    r.add_argument("--endpoint")
    r.add_argument("--api-model", dest="api_model")
    r.add_argument("--from-config", action="store_true")
    r.add_argument("--limit", type=int)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("score", help="score raw outputs vs references")
    s.add_argument("model_id")
    s.add_argument("--lang", required=True, help="test-set stem, e.g. afr or afr-post")
    s.set_defaults(func=cmd_score)

    rp = sub.add_parser("report", help="generate the HTML eval report (+ console table)")
    rp.add_argument("--text", action="store_true", help="console table only, skip HTML")
    rp.add_argument("--date", help="date string for the report (default: today)")
    rp.add_argument("--title")
    rp.set_defaults(func=cmd_report)

    cm = sub.add_parser("comet", help="add COMET semantic scores to scores.json (no re-runs)")
    cm.add_argument("--models", nargs="*", help="restrict to these model ids")
    cm.add_argument("--langs", nargs="*", help="restrict to these test-set stems (e.g. afr-pre afr-post)")
    cm.set_defaults(func=cmd_comet)

    sw = sub.add_parser("sweep", help="run+score on-host models (have:true) over languages")
    sw.add_argument("--models", nargs="*", help="restrict to these model ids (default: all have:true)")
    sw.add_argument("--langs", nargs="*", help="languages (default: afr deu spa)")
    sw.add_argument("--limit", type=int)
    sw.set_defaults(func=cmd_sweep)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
