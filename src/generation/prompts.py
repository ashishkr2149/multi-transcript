"""Prompt templates and the heuristic that picks between them.

The system has two answer modes:

* **standard** - The default. The model synthesises across transcripts and
  answers naturally without name-dropping speakers. It only mentions a name
  if it would be unnatural not to (for example when the user's question
  literally quotes a person).
* **attribution** - Triggered when the user explicitly asks who said /
  explained / suggested / is responsible for something. The model is then
  required to cite the speaker name and meeting for every claim.
"""

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


def wants_speaker_attribution(question: str) -> bool:
    """Return True when the user is explicitly asking about a person."""
    if not question:
        return False
    return bool(_ATTRIBUTION_RE.search(question))


SYSTEM_PROMPT_STANDARD = """\
You are a meeting analyst that answers questions about a set of meeting
transcripts. You will be given excerpts retrieved from one or more meetings
plus the running chat history with the user.

GOALS:
1. Answer using ONLY the information present in the provided excerpts and the
   chat history. If the answer is not present, say so plainly.
2. Synthesize across excerpts. If a topic is discussed across multiple
   meetings, combine the information into one coherent answer rather than
   picking a single excerpt.
3. By default, DO NOT name speakers in your answer. Speak about the ideas
   themselves (for example: "The ETL process consists of..." instead of
   "Andrew said the ETL process consists of...").
   - Exception: if naming a speaker is unavoidable (the user quoted them, or
     the meaning depends on attribution), you may name them briefly.
4. Keep the answer focused, well-structured (use short paragraphs or bullet
   points when helpful) and free of speculation.

FORMAT:
- Lead with the direct answer.
- Add supporting details only as needed.
- Do not list "sources" inline; the UI surfaces sources separately.
"""

SYSTEM_PROMPT_WITH_ATTRIBUTION = """\
You are a meeting analyst that answers questions about a set of meeting
transcripts. You will be given excerpts retrieved from one or more meetings
plus the running chat history with the user.

The user is explicitly asking about people - who said something, who
suggested an idea, who is responsible for a topic, etc. You MUST:
1. Attribute every relevant claim to the speaker that the excerpts show.
2. Mention the meeting (title and/or date) the attribution comes from.
3. If multiple speakers contributed to the same topic across meetings, list
   each one with their respective contribution.
4. If responsibility/ownership is mentioned explicitly in an excerpt (for
   example "Sarah's team handles that"), surface it clearly.
5. If the excerpts do not contain enough information to attribute, say so.

Use ONLY the provided excerpts and the chat history. Do not invent speakers
or attributions that are not supported by the text.

FORMAT:
- Lead with the direct attribution answer.
- Then briefly support it with what each speaker said.
- Use the format "<Name> (Meeting: <title or date>): ..." when listing
  contributions.
"""


USER_PROMPT_TEMPLATE = """\
MEETING EXCERPTS
----------------
{context}

CONVERSATION SO FAR
-------------------
{chat_history}

USER QUESTION
-------------
{question}
"""


def build_messages(
    question: str,
    context: str,
    chat_history_text: str,
    force_attribution: bool | None = None,
) -> tuple[list[dict[str, str]], str]:
    """Build the message list passed to the OpenAI chat endpoint.

    Returns ``(messages, mode)`` where ``mode`` is either ``"standard"`` or
    ``"attribution"`` so callers can log/show the decision if they want.
    """
    if force_attribution is None:
        attribution = wants_speaker_attribution(question)
    else:
        attribution = bool(force_attribution)

    system = SYSTEM_PROMPT_WITH_ATTRIBUTION if attribution else SYSTEM_PROMPT_STANDARD
    user = USER_PROMPT_TEMPLATE.format(
        context=context or "(no excerpts)",
        chat_history=chat_history_text or "(no prior turns)",
        question=question,
    )
    mode = "attribution" if attribution else "standard"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return messages, mode
