"""The translation prompt — held CONSTANT across every model for fairness.

We send a SINGLE user message (no `system` role). Reasons:
- Some chat templates (notably Gemma) reject a `system` role and 400.
- A single message is byte-for-byte identical across models, so no model gets a
  structural advantage. Consistency matters more than per-model prompt tuning in
  a benchmark.

IMPORTANT methodology note. This is a generic, minimal sentence-level MT prompt.
It is intentionally NOT Lector's in-app prompt: Lector translates a *clicked word
or phrase* and asks for a structured JSON dictionary entry (senses, IPA,
etymology, idiom/register notes). That task has no canonical gold output, so it
can't be scored against Tatoeba. This harness therefore measures **general
source->English sentence MT quality as a proxy** for translation-agent quality.
See README "Methodology".
"""

INSTRUCTION = (
    "You are a professional translator. Translate the following {language} "
    "sentence into natural, fluent English.\n\n"
    "{language} sentence:\n{source}"
)


def build(source_text: str, language_name: str) -> str:
    """Return the single user-message content for one source sentence."""
    return INSTRUCTION.format(language=language_name, source=source_text)
