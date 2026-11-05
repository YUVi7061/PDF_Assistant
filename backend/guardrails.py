"""Input / context / output guardrails.

Layer 1 (always on, offline): length, prompt-injection, blocked topics, PII + secret redaction,
                              system-prompt leakage, lexical grounding check.
Layer 2 (optional): Amazon Bedrock Guardrails via ApplyGuardrail when BEDROCK_GUARDRAIL_ID is set.
"""
import re
from dataclasses import dataclass, field

from . import config

SYSTEM_PROMPT_MARKER = "PDF-ASSISTANT-SYSTEM-RULES"

_INJECTION = [
    r"ignore (all |any |the )?(previous|prior|above|earlier|preceding) (instructions|prompts?|rules|context)",
    r"disregard (all |any |the )?(previous|prior|above|earlier|preceding)",
    r"forget (everything|all) (you|that) (were|was|have been) (told|instructed)",
    r"(reveal|show|print|repeat|output|display|leak) (me )?(your|the) (system|hidden|initial|original) (prompt|instructions|rules)",
    r"you are now (in )?(dan|developer mode|jailbroken|unrestricted)",
    r"act as (an? )?(unrestricted|unfiltered|jailbroken)",
    r"\bjailbreak\b",
    r"override (your|the) (safety|guardrails?|instructions|rules)",
    r"<\s*/?\s*(system|assistant)\s*>",
    r"\[\s*/?\s*(inst|system)\s*\]",
]
INJECTION_RE = [re.compile(p, re.I) for p in _INJECTION]

_BLOCKED_TOPICS = [
    r"how (do i|to|can i) (make|build|synthesi[sz]e) (a )?(bomb|explosive|nerve agent|bioweapon)",
    r"write (me )?(a )?(working )?(malware|ransomware|keylogger|computer virus)",
]
BLOCKED_TOPIC_RE = [re.compile(p, re.I) for p in _BLOCKED_TOPICS]

_PII = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE": re.compile(r"(?<!\w)(?:\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\w)|\+\d{1,3}[\s-]?\d{10}(?!\d)"),
    "AADHAAR": re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"),
    "PAN": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "AWS_KEY": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "SECRET": re.compile(r"(?i)\b(?:api[_-]?key|secret|password|token)\s*[:=]\s*\S{6,}"),
}
_CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")


@dataclass
class GuardResult:
    allowed: bool = True
    text: str = ""
    flags: list[str] = field(default_factory=list)
    message: str = ""


def _luhn(num: str) -> bool:
    digits = [int(c) for c in num if c.isdigit()]
    if not 13 <= len(digits) <= 16:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    found: list[str] = []

    def card(m):
        if _luhn(m.group(0)):
            found.append("CREDIT_CARD")
            return "[REDACTED CREDIT_CARD]"
        return m.group(0)

    text = _CARD.sub(card, text)  # first, so a card number isn't mis-labelled as Aadhaar/phone
    for label, rx in _PII.items():
        if rx.search(text):
            found.append(label)
            text = rx.sub(f"[REDACTED {label}]", text)
    return text, sorted(set(found))


# ------------------------------------------------------------------ input
def check_input(text: str, bedrock=None) -> GuardResult:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text or "").strip()
    if not text:
        return GuardResult(False, text, ["EMPTY"], "Please enter a question.")
    if len(text) > config.MAX_QUESTION_CHARS:
        return GuardResult(False, text, ["TOO_LONG"], f"Question too long (max {config.MAX_QUESTION_CHARS} characters).")
    if any(rx.search(text) for rx in INJECTION_RE):
        return GuardResult(False, text, ["PROMPT_INJECTION"], "I can't follow instructions that try to change my rules. Ask me a question about the document instead.")
    if any(rx.search(text) for rx in BLOCKED_TOPIC_RE):
        return GuardResult(False, text, ["BLOCKED_TOPIC"], "I can't help with that request.")
    if bedrock is not None and config.GUARDRAIL_ID:
        try:
            intervened, out = bedrock.apply_guardrail(text, "INPUT")
            if intervened:
                return GuardResult(False, text, ["BEDROCK_GUARDRAIL_INPUT"], out)
        except Exception as e:  # noqa: BLE001  fail closed
            return GuardResult(False, text, ["BEDROCK_GUARDRAIL_ERROR"], f"Safety check unavailable: {e}")
    return GuardResult(True, text)


# ------------------------------------------------------------------ retrieved context (indirect injection)
def sanitize_context(chunks: list[dict]) -> tuple[list[dict], list[str]]:
    kept, flags = [], []
    for c in chunks:
        if any(rx.search(c["text"]) for rx in INJECTION_RE):
            flags.append(f"CONTEXT_INJECTION_DROPPED(p{c['page']})")
        else:
            kept.append(c)
    return kept, flags


# ------------------------------------------------------------------ output
_STOP = set("this that with from have been were will would there their which about into than then them they what when where your also such more some only other these those".split())


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 3 and w not in _STOP}


def grounding_score(answer: str, context: str) -> float:
    a = _tokens(answer)
    return 1.0 if not a else len(a & _tokens(context)) / len(a)


def check_output(answer: str, context: str, bedrock=None) -> GuardResult:
    flags: list[str] = []
    if SYSTEM_PROMPT_MARKER in answer:
        return GuardResult(False, config.NOT_FOUND_MSG, ["SYSTEM_PROMPT_LEAK"], config.NOT_FOUND_MSG)

    if config.REDACT_OUTPUT_PII:
        answer, found = redact_pii(answer)
        flags += [f"PII_REDACTED:{f}" for f in found]

    if bedrock is not None and config.GUARDRAIL_ID:
        try:
            intervened, out = bedrock.apply_guardrail(answer, "OUTPUT")
            if intervened:
                return GuardResult(False, out, flags + ["BEDROCK_GUARDRAIL_OUTPUT"], out)
        except Exception as e:  # noqa: BLE001
            flags.append(f"BEDROCK_GUARDRAIL_ERROR:{type(e).__name__}")

    if config.NOT_FOUND_MSG not in answer and context:
        score = grounding_score(answer, context)
        if score < 0.25:
            flags.append(f"LOW_GROUNDING:{score:.2f}")
            answer += "\n\n⚠️ _This answer may not be well supported by the document. Please verify against the cited pages._"
    return GuardResult(True, answer, flags)
