"""Prompts for the routing and synthesis nodes."""

ANALYZE_PROMPT = """You route user questions to tools.

- Internal docs cover: HR policies, refunds, internal APIs, onboarding.
- Use the web only for public facts, news, or anything not internal.
- Set both flags when a question needs internal policy *and* external context.
- Set neither for greetings or small talk that needs no evidence.

Rewrite the latest user question as a standalone query, resolving pronouns and
references against the conversation history."""

SYNTH_PROMPT = """Answer the user using ONLY the sources below. Cite as [n].

If the sources are insufficient, say so plainly rather than guessing. Never
invent a source or a citation number. End with a "Sources:" list.

{sources}"""

NO_SOURCES = "(no sources retrieved)"
