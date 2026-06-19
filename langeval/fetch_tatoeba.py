"""Build blinded source->English test sets from Tatoeba.

For a language code (ISO 639-3: afr/deu/spa) we download Tatoeba exports, join
source sentences to ALL of their linked English translations (multi-ref), apply
length/dedup filters, take a seeded random sample, and write a test set.

We use the *detailed* source export, which carries `date_added` per sentence.
That lets us build date-bucketed sets (--after / --before) for a contamination
check: compare a model on pairs that predate its training vs pairs provably added
afterwards. If scores fall apart on the post-cutoff bucket, that's memorisation.

The test set stores the English references, but the runner only reads `source` —
that is the blinding.
"""

import bz2
import json
import random
from pathlib import Path

import requests

TATOEBA = "https://downloads.tatoeba.org/exports/per_language"
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
TESTSETS = ROOT / "data" / "testsets"

LANG_NAMES = {"afr": "Afrikaans", "deu": "German", "spa": "Spanish"}


def base_lang(stem: str) -> str:
    """'afr-post' -> 'afr'. Test-set stems may carry a variant suffix."""
    return stem.split("-", 1)[0]


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached  {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetch   {url}", flush=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    return dest


def _read_detailed(bz2_path: Path) -> dict:
    """{lang}_sentences_detailed.tsv -> {id: (text, date)}.
    Columns: id, lang, text, username, date_added, date_last_modified."""
    out = {}
    with bz2.open(bz2_path, "rt", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                date = p[4][:10] if len(p) >= 5 and p[4][:4].isdigit() else None
                out[p[0]] = (p[2], date)
    return out


def _read_sentences(bz2_path: Path) -> dict:
    """Plain {lang}_sentences.tsv -> {id: text}.  Columns: id, lang, text."""
    out = {}
    with bz2.open(bz2_path, "rt", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                out[p[0]] = p[2]
    return out


def _read_links(bz2_path: Path) -> list:
    pairs = []
    with bz2.open(bz2_path, "rt", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                pairs.append((p[0], p[1]))
    return pairs


def build(lang: str, n: int = 200, seed: int = 13, min_words: int = 4,
          max_words: int = 25, after: str | None = None, before: str | None = None,
          tag: str | None = None) -> Path:
    """Build a test set. after/before are 'YYYY-MM-DD' filters on date_added
    (ISO dates sort lexically). tag names the output variant, e.g. tag='post'
    -> data/testsets/afr-post.json."""
    name = LANG_NAMES.get(lang, lang)
    src_f = _download(f"{TATOEBA}/{lang}/{lang}_sentences_detailed.tsv.bz2",
                      RAW / f"{lang}_sentences_detailed.tsv.bz2")
    lnk_f = _download(f"{TATOEBA}/{lang}/{lang}-eng_links.tsv.bz2",
                      RAW / f"{lang}-eng_links.tsv.bz2")
    eng_f = _download(f"{TATOEBA}/eng/eng_sentences.tsv.bz2",
                      RAW / "eng_sentences.tsv.bz2")

    print("  parsing source (dated) + english + links ...", flush=True)
    src = _read_detailed(src_f)          # id -> (text, date)
    eng = _read_sentences(eng_f)         # id -> text
    links = _read_links(lnk_f)

    refs: dict = {}
    for a, b in links:
        if a in src and b in eng:
            refs.setdefault(a, []).append(eng[b])
        elif b in src and a in eng:
            refs.setdefault(b, []).append(eng[a])

    seen = set()
    cands = []
    for sid, (text, date) in src.items():
        if sid not in refs:
            continue
        if (after and (not date or date < after)) or (before and (not date or date >= before)):
            continue
        wc = len(text.split())
        if wc < min_words or wc > max_words:
            continue
        key = text.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        cands.append({"id": sid, "source": text, "date": date,
                      "refs": sorted(set(refs[sid]))})

    rng = random.Random(seed)
    rng.shuffle(cands)
    sample = cands[:n]

    TESTSETS.mkdir(parents=True, exist_ok=True)
    stem = f"{lang}-{tag}" if tag else lang
    out = TESTSETS / f"{stem}.json"
    dates = sorted(c["date"] for c in sample if c["date"])
    meta = {
        "lang": lang, "language_name": name, "stem": stem,
        "requested": n, "available": len(cands), "selected": len(sample),
        "seed": seed, "filters": {"min_words": min_words, "max_words": max_words,
                                   "after": after, "before": before},
        "date_range": [dates[0], dates[-1]] if dates else None,
        "multi_ref": True, "source_corpus": "Tatoeba", "license": "CC-BY 2.0 FR",
    }
    out.write_text(json.dumps({"meta": meta, "items": sample},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    avg_refs = sum(len(i["refs"]) for i in sample) / max(len(sample), 1)
    drange = f"{dates[0]}..{dates[-1]}" if dates else "n/a"
    print(f"  wrote {out}  ({len(sample)} items, {avg_refs:.1f} refs/item, "
          f"dates {drange}, {len(cands)} available)")
    return out
