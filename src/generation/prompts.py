"""Prompt templates and the heuristic that picks between them."""

from __future__ import annotations

import re

ATTRIBUTION_KEYWORDS = [
    r"\bwho\s+(said|says|spoke|mentioned|explained|suggested|asked|told|shared|brought|raised|proposed|wanted|claimed|noted|owns|owned|leads|leads?\b|leading|responsible)\b",
    r"\bwhich\s+(person|speaker|participant|attendee)\b",
    r"\bby\s+whom\b",
    r"\bresponsible\s+for\b",
    r"\bin\s+charge\s+of\b",
    r"\bowner\s+of\b",
    r"\bspeaker(s)?\s+(of|for|behind)\b",
    r"\battribut(e|ion|ed)\b",
    r"\bcredit\s+(for|to)\b",
    r"\baccording\s+to\s+whom\b",
]
_ATTRIBUTION_RE = re.compile("|".join(ATTRIBUTION_KEYWORDS), re.IGNORECASE)

_GROUNDING_RULES = """
ACCURACY RULES (critical):
- When INDEXED MEETINGS is present, use it as the authoritative source for how many
  meetings exist, their dates, and transcript_id values.
- Never count excerpts, bullet points, or sections you write as separate meetings.
- Multiple excerpts with the same transcript_id or meeting date are ONE meeting.
- Do not add "additional excerpt" sections — one summary block per indexed meeting.
- Do not state facts absent from INDEXED MEETINGS or MEETING EXCERPTS. If unsure, say so.
"""

_INVENTORY_RULES = """
INVENTORY RULES:
- The meeting count must equal the number of entries in INDEXED MEETINGS.
- List each meeting date exactly once.
"""

_PER_MEETING_RULES = """
PER-MEETING SUMMARY RULES:
- Produce exactly one section per meeting in INDEXED MEETINGS (same count).
- Label each section with the meeting title and date from INDEXED MEETINGS.
"""


def wants_speaker_attribution(question: str) -> bool:
    """Return True when the user is explicitly asking about a person."""
    if not question:
        return False
    return bool(_ATTRIBUTION_RE.search(question))


SYSTEM_PROMPT_STANDARD = """\
You are a meeting analyst that answers questions about a set of meeting
transcripts. You will be given an INDEXED MEETINGS catalog (authoritative metadata)
and MEETING EXCERPTS (evidence of what was said), plus chat history.

GOALS:
1. Answer using ONLY the information present in INDEXED MEETINGS, EXCERPTS, and
   chat history. If the answer is not present, say so plainly.
2. Synthesize across excerpts. If a topic is discussed across multiple
   meetings, combine the information into one coherent answer rather than
   picking a single excerpt.
3. By default, DO NOT name speakers in your answer. Speak about the ideas
   themselves (for example: "The ETL process consists of..." instead of
   "Andrew said the ETL process consists of...").
   - Exception: if naming a speaker is unavoidable, you may name them briefly.
4. Keep the answer focused, well-structured (use short paragraphs or bullet
   points when helpful) and free of speculation.
""" + _GROUNDING_RULES + """
FORMAT:
- Lead with the direct answer.
- Add supporting details only as needed.
- Do not list "sources" inline; the UI surfaces sources separately.
"""

SYSTEM_PROMPT_WITH_ATTRIBUTION = """\
You are a meeting analyst that answers questions about a set of meeting
transcripts. You will be given INDEXED MEETINGS and MEETING EXCERPTS plus chat history.

The user is explicitly asking about people - who said something, who
suggested an idea, who is responsible for a topic, etc. You MUST:
1. Attribute every relevant claim to the speaker that the excerpts show.
2. Mention the meeting (title and/or date) and transcript_id when citing.
3. If multiple speakers contributed to the same topic across meetings, list
   each one with their respective contribution.
4. If the excerpts do not contain enough information to attribute, say so.

Use ONLY the provided INDEXED MEETINGS, EXCERPTS, and chat history.
""" + _GROUNDING_RULES + """
FORMAT:
- Lead with the direct attribution answer.
- Use "<Name> (Meeting: <title or date>): ..." when listing contributions.
"""

USER_PROMPT_TEMPLATE = """\
CONTEXT
-------
{context}

CONVERSATION SO FAR
-------------------
{chat_history}

USER QUESTION
-------------
{question}
"""


def _intent_rules(intent: str | None) -> str:
    if intent == "inventory":
        return _INVENTORY_RULES
    if intent == "per_meeting_summary":
        return _PER_MEETING_RULES
    return ""


def build_messages(
    question: str,
    context: str,
    chat_history_text: str,
    force_attribution: bool | None = None,
    intent: str | None = None,
    inventory_preface: str | None = None,
) -> tuple[list[dict[str, str]], str]:
    """Build the message list passed to the OpenAI chat endpoint."""
    if force_attribution is None:
        attribution = wants_speaker_attribution(question)
    else:
        attribution = bool(force_attribution)

    system = SYSTEM_PROMPT_WITH_ATTRIBUTION if attribution else SYSTEM_PROMPT_STANDARD
    system += _intent_rules(intent)
    if inventory_preface:
        system += f"\n\nFACTUAL PREFACE (must align your answer with this):\n{inventory_preface}\n"

    user = USER_PROMPT_TEMPLATE.format(
        context=context or "(no context)",
        chat_history=chat_history_text or "(no prior turns)",
        question=question,
    )
    mode = "attribution" if attribution else "standard"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return messages, mode
