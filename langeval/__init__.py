"""llm-lang-eval — a small, reproducible harness for benchmarking how well
local (and cloud) LLMs translate source languages into English.

Direction: source (Afrikaans / German / Spanish) -> English.
References: Tatoeba sentence pairs (multi-reference where available).
Metrics: chrF++ and BLEU via sacreBLEU; COMET optional.
"""

__version__ = "0.1.0"
