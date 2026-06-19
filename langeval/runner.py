"""Run one model over a blinded test set via an OpenAI-compatible chat endpoint.

Design constraints (see README):
- One model at a time. On an 18 GB host you cannot hold several models; rely on
  the server's JIT load and run models sequentially.
- Deterministic decoding: temperature 0.
- No chain-of-thought. reasoning_effort=none disables the reasoning channel.
- STRUCTURED OUTPUT. We constrain every model to emit {"translation": "..."} via
  response_format json_schema. This is the great equaliser: small/chatty models
  (e.g. Gemma 4 E4B "thinks out loud" in plain text) otherwise bury the answer in
  preamble and get mis-scored. Constrained decoding makes that impossible, gives
  every model the identical constraint, and mirrors how Lector itself prompts.
- Persist raw output + finish_reason + reasoning tokens for auditability.
"""

import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_OUT = ROOT / "results" / "raw_outputs"

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

TRANSLATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "translation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"translation": {"type": "string"}},
            "required": ["translation"],
            "additionalProperties": False,
        },
    },
}


class TranslateError(RuntimeError):
    """Carries the server's response body so failures are diagnosable."""


def _text_fallback(raw: str) -> str:
    """If a model ignores the schema (e.g. on-device Apfel), recover the
    translation: prefer the last quoted span, else strip preambles off the
    last line."""
    txt = THINK_RE.sub("", raw or "").strip()
    quotes = re.findall(r'["“]([^"”]{2,})["”]', txt)
    if quotes:
        return quotes[-1].strip()
    line = next((ln.strip() for ln in reversed(txt.splitlines()) if ln.strip()), txt)
    line = re.sub(r"(?i)^(here(?:'s| is) )?(the )?(english )?translation"
                  r"(?: of the \w+ sentence)?(?: is| would be)?\s*[:\-]?\s*", "", line)
    line = re.sub(r"(?i)^the sentence translates to\s*", "", line)
    return line.strip().strip('"').strip("“”").strip()


def parse_translation(raw: str) -> str:
    """Pull the translation from the JSON-schema response; fall back to text."""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("translation"), str):
            return obj["translation"].strip()
    except (json.JSONDecodeError, TypeError):
        pass
    return _text_fallback(raw)


def translate_one(endpoint: str, api_model: str, prompt: str,
                  timeout: int = 180, extra_body: dict | None = None,
                  no_schema: bool = False, api_key: str | None = None) -> dict:
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": api_model,
        "temperature": 0,
        "max_tokens": 512,
        "messages": [{"role": "user", "content": prompt}],
    }
    if not no_schema:  # constrained models: reasoning off + JSON schema
        body["reasoning_effort"] = "none"
        body["response_format"] = TRANSLATION_SCHEMA
    if extra_body:
        body.update(extra_body)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        r = requests.post(url, json=body, timeout=timeout, headers=headers)
    except requests.RequestException as e:
        raise TranslateError(f"request failed: {e}") from e
    if not r.ok:
        detail = r.text[:400]
        try:
            detail = r.json().get("error", {}).get("message", detail)
        except Exception:
            pass
        raise TranslateError(f"HTTP {r.status_code}: {detail}")
    j = r.json()
    ch = j["choices"][0]
    usage = j.get("usage", {}) or {}
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    return {
        "content": ch["message"].get("content") or "",
        "finish_reason": ch.get("finish_reason"),
        "reasoning_tokens": reasoning,
        "completion_tokens": usage.get("completion_tokens"),
    }


def preflight(endpoint: str, api_model: str, language_name: str,
              extra_body: dict | None = None, no_schema: bool = False,
              api_key: str | None = None) -> None:
    """One probe. Raises if the model can't serve OR yields no parseable
    translation — so the run fails fast with a reason instead of zeros."""
    from . import prompts
    res = translate_one(endpoint, api_model,
                        prompts.build("Hallo, hoe gaan dit met jou?", language_name),
                        extra_body=extra_body, no_schema=no_schema, api_key=api_key)
    if not parse_translation(res.get("content") or "").strip():
        raise TranslateError(
            f"no parseable translation (finish={res.get('finish_reason')}, "
            f"reasoning_tokens={res.get('reasoning_tokens')}, "
            f"raw={(res.get('content') or '')[:120]!r})")


def run(model_id: str, endpoint: str, api_model: str, testset_path: Path,
        language_name: str, prompt_fn, limit: int | None = None,
        extra_body: dict | None = None, no_schema: bool = False,
        api_key: str | None = None) -> Path:
    items = json.loads(Path(testset_path).read_text(encoding="utf-8"))["items"]
    if limit:
        items = items[:limit]
    RAW_OUT.mkdir(parents=True, exist_ok=True)
    lang = Path(testset_path).stem
    out = RAW_OUT / f"{model_id}__{lang}.jsonl"

    n_err = n_empty = 0
    with open(out, "w", encoding="utf-8") as f:
        for i, it in enumerate(items, 1):
            prompt = prompt_fn(it["source"], language_name)
            try:
                t0 = time.time()
                res = translate_one(endpoint, api_model, prompt, extra_body=extra_body,
                                    no_schema=no_schema, api_key=api_key)
                hyp = parse_translation(res["content"])
                rec = {
                    "id": it["id"], "source": it["source"],
                    "raw": res["content"], "hypothesis": hyp,
                    "finish_reason": res["finish_reason"],
                    "reasoning_tokens": res["reasoning_tokens"],
                    "latency_s": round(time.time() - t0, 2),
                }
                if not hyp:
                    n_empty += 1
                    rec["warn"] = "empty"
                tag = hyp[:70] if hyp else f"[EMPTY finish={res['finish_reason']}]"
            except Exception as e:
                n_err += 1
                rec = {"id": it["id"], "source": it["source"], "error": str(e)[:200]}
                tag = "ERROR: " + rec["error"][:60]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{i:>3}/{len(items)}] {model_id:<16} {tag}")
    print(f"  -> {out}  ({len(items) - n_err - n_empty} ok, {n_empty} empty, {n_err} errors)")
    return out
