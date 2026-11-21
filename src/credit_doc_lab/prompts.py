"""Prompt text for the baseline extractor. Bump PROMPT_VERSION when editing."""

from .config import PROMPT_VERSION  # noqa: F401  (re-exported for run records)

SYSTEM_INSTRUCTION = """\
You extract structured loan terms from a credit agreement.
Use only the attached document. Do not use outside knowledge.
Fill every field defined by the response schema, following each field's description.
If the document does not state a value (and it cannot be derived as the field description allows), return null; never guess.
Copy names exactly as written in the document."""

USER_INSTRUCTION = "Extract the fields defined by the response schema from the attached credit agreement."
