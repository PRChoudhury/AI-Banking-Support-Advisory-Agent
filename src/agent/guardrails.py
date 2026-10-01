"""
Input guardrails for the banking agent.
"""
import re
# ==========================================================
# Masking Function
# ==========================================================

_LABEL_TAG = {
    "aadhaar": "[ID]", "aadhar": "[ID]", "adhaar": "[ID]", "adhar": "[ID]", "uid": "[ID]",
    "account": "[ACCOUNT]", "a/c": "[ACCOUNT]", "acct": "[ACCOUNT]",
    "cif": "[ACCOUNT]", "customer id": "[ACCOUNT]",
    "card": "[CARD]", "debit card": "[CARD]", "credit card": "[CARD]",
    "cvv": "[CVV]", "cvc": "[CVV]", "pin": "[PIN]", "otp": "[OTP]", "pan": "[PAN]",
    "phone": "[PHONE]", "mobile": "[PHONE]", "contact": "[PHONE]", "whatsapp": "[PHONE]",
}

LABELLED_PII_RE = re.compile(
    r"\b(aadhaar|aadhar|adhaar|adhar|uid"
    r"|debit\s*card|credit\s*card|card"
    r"|account|a/c|acct|cif|customer\s*id"
    r"|cvv|cvc|otp|pin|pan"
    r"|phone|mobile|whatsapp|contact)\b"
    r"\s*(?:number|no\.?|num|id)?\s*(?:is|are|=|:|-)?\s*"
    r"([0-9][0-9\s\-]{1,22}[0-9]|[A-Za-z]{5}[0-9]{4}[A-Za-z])",
    re.I,
)
CITATION_RE = re.compile(r"\([\w.\-]+\.pdf,\s*p\.\d+\)", re.I)

OUT_OF_SCOPE_RE = re.compile(
    r"\b(transfer|send money|pay my|block (my )?card|unblock"
    r"|open (an? )?account|close my account|change my (address|number))\b", re.I)

REFUSAL_RE = re.compile(
    r"(i'?m not able to|i can'?t|i cannot|i am unable|unable to|not permitted"
    r"|has to go through|in the .{0,25}section (of )?the app)", re.I)

PERSONAL_ADVICE_RE = re.compile(
    r"\bshould i\b|\bwhich is better\b|\brecommend\b.{0,25}\b(me|my|i should)\b"
    r"|\bmy situation\b", re.I)

DISCLAIMER_RE = re.compile(
    r"\badvisor\b"
    r"|general (education|information) only\b|not (a |)personalized\b", re.I)

DISCLAIMER_TEXT = ("DISCLAIMER : No personalized financial/investment advice — "
                    "general education only.")

def _tag_for(label: str) -> str:
    return _LABEL_TAG.get(re.sub(r"\s+", " ", label).strip().lower(), "[REDACTED]")


def mask_private_data(text: str) -> str:
    """
    Detects and masks sensitive information like account and card numbers,
    phone numbers, emails, PAN, Aadhaar, CVV and OTP, before LLM processing.
    """
    if not text:
        return text

    text = LABELLED_PII_RE.sub(
        lambda m: f"{m.group(1)} {_tag_for(m.group(1))}", text)

    # 2. Email addresses
    text = re.sub(
        r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
        "[EMAIL]", text, flags=re.IGNORECASE)

    text = re.sub(r'\b(?:\d{4}[ -]?){3}\d{4}\b|\b\d{13,19}\b', "[CARD]", text)

    # 4. Indian PAN format, either case
    text = re.sub(r'\b[A-Z]{5}\d{4}[A-Z]\b', "[PAN]", text, flags=re.IGNORECASE)

    # 5. Aadhaar-like 12-digit sequences, with or without spacing
    text = re.sub(r'\b\d{4}[ -]?\d{4}[ -]?\d{4}\b', "[ID]", text)

    # 6. Indian mobile, tolerant of +91 and one inner separator. The generic
    #    3-3-4 pattern is gone: it masked "Invoice 123 456 7890".
    text = re.sub(r'(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)', "[PHONE]", text)

    return text


# ==========================================================
# Pre-processing Layer
# ==========================================================
def sanitize_and_respond(user_input: str) -> str:
    """
    Pre-processing layer that sanitizes all incoming user text
    before passing to the model.
    """
    return mask_private_data(user_input)

def citation_ok(answer: str, sources: list[dict]) -> bool:
    """
    True if the answer cites a source whenever kb_lookup actually returned one.
    `sources` is the list from collect_sources() — {"source": ..., "page": ...} dicts.
    Numbers from emi_calculator alone (no kb_lookup) are exempt, since only the
    rate feeding the calculation needs a citation, not the computed EMI figure.
    """
    if not sources:
        return True
    return bool(CITATION_RE.search(answer or ""))

def disclaimer_ok(question, answer):
    return not PERSONAL_ADVICE_RE.search(question or "") or bool(DISCLAIMER_RE.search(answer or ""))

def needs_disclaimer(question: str) -> bool:
    return bool(PERSONAL_ADVICE_RE.search(question or ""))
