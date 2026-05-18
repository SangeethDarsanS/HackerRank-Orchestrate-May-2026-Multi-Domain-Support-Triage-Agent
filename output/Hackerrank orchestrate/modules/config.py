"""
Configuration module for the Support Triage Agent.
All constants, paths, thresholds, and keyword lists live here.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
# code/ lives one level under the repo root
CODE_DIR = Path(__file__).resolve().parent.parent       # code/
BASE_DIR = CODE_DIR.parent                              # repo root
DATA_DIR = BASE_DIR / "data"
SUPPORT_TICKETS_DIR = BASE_DIR / "support_tickets"
CACHE_DIR = CODE_DIR / ".cache"

# ---------------------------------------------------------------------------
# Output files (written by main.py)
# ---------------------------------------------------------------------------
OUTPUT_CSV = SUPPORT_TICKETS_DIR / "output.csv"
TRIAGE_LOG_FILE = SUPPORT_TICKETS_DIR / "log.txt"

# ---------------------------------------------------------------------------
# AGENTS.md log (per §2 of AGENTS.md)
# ---------------------------------------------------------------------------
AGENTS_LOG_DIR = Path.home() / "hackerrank_orchestrate"
AGENTS_LOG_FILE = AGENTS_LOG_DIR / "log.txt"

# ---------------------------------------------------------------------------
# Embedding / retrieval
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384          # dimensionality for all-MiniLM-L6-v2
TOP_K_CHUNKS = 10            # chunks returned from FAISS per query
TOP_K_DOCS = 5               # unique documents to surface after dedup
SIMILARITY_THRESHOLD = 0.20  # cosine sim floor; below → no usable match
CONFIDENCE_THRESHOLD = 0.28  # overall confidence floor; below → escalate

# Deterministic seed (for anything random in the pipeline)
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Domains
# ---------------------------------------------------------------------------
DOMAINS = ["hackerrank", "claude", "visa"]

# Keyword signals used when Company field is absent / "None"
DOMAIN_KEYWORDS = {
    "hackerrank": [
        "hackerrank", "hacker rank", "test", "assessment", "candidate",
        "coding", "screen", "interview", "recruiter", "developer",
        "mock interview", "library", "skillup", "chakra", "engage",
        "certificate", "resume builder",
    ],
    "claude": [
        "claude", "anthropic", "ai model", "llm", "conversation", "chat",
        "claude.ai", "bedrock", "claude api", "prompt", "token", "claude pro",
        "claude team", "workspace", "lti", "claude code",
    ],
    "visa": [
        "visa", "card", "payment", "bank", "merchant", "transaction",
        "atm", "pin", "cheque", "refund", "dispute", "chargeback",
        "traveller", "visa card", "credit card", "debit card",
    ],
}

# ---------------------------------------------------------------------------
# Request type classification
# ---------------------------------------------------------------------------
REQUEST_TYPES = ["product_issue", "feature_request", "bug", "invalid"]

# Patterns that map text → request type
# Checked in order; first match wins; default = product_issue
INVALID_PATTERNS = [
    r"iron\s+man",
    r"\bactor\b.*\bmovie\b",
    # Gratitude / closing statements (not support requests)
    r"^(?:thanks|thank\s+you|ty)[\s.!,]*(?:for\s+\w+ing?\s+me[\s.!]*)?$",
    r"^(?:hi|hello|hey)[\s.!,]*(?:there[\s.!,]*)?$",
    r"^(?:thanks|thank\s+you)\s+for\s+(?:your\s+)?(?:help|helping|assistance|support)\s*[.!]?\s*$",
    r"^(?:you'?re?\s+)?welcome\s*[.!]?\s*$",
    r"give\s+me\s+(?:the\s+)?code\s+to\s+delete",
    r"delete\s+all\s+files",
    r"\brm\s+-rf\b",
    r"ignore\s+(?:previous|prior|above)\s+instructions",
    r"(?:affiche|montre|révèle|show|reveal)\s+(?:toutes?\s+les?\s+)?(?:règles|logique|instructions?|prompt|system)",
    r"règles\s+internes",
    r"logique\s+exacte",
    r"documents\s+récupérés",
    r"disregard\s+(?:previous|above)\s+instructions",
    r"you\s+are\s+now\s+(?:a|an)\s+(?!support|helpful)",
    r"forget\s+(?:everything|all)\s+(?:you\s+)?(?:know|were|have)",
    r"new\s+persona",
    r"act\s+as\s+(?:a|an)\s+(?!support|helpful)",
]

BUG_PATTERNS = [
    r"(?:is|are|has|have)\s+(?:been\s+)?(?:down|broken|crashing|unavailable)",
    r"(?:not|isn'?t|won'?t|doesn'?t|can'?t)\s+(?:work|load|open|access|connect|respond|function)",
    r"\berror\b(?:\s+message)?",
    r"\bcrash(?:ing|ed)?\b",
    r"\bfail(?:ing|ed|ure)?\b",
    r"\boutage\b",
    r"\b500\s+error\b",
    r"(?:none|no)\s+of\s+the\s+(?:pages?|submissions?|challenges?|requests?)\s+(?:are\s+)?(?:work|load|access)",
    r"\bbug\b",
    r"\bnot\s+(?:working|responding|loading|functioning)\b",
]

FEATURE_REQUEST_PATTERNS = [
    r"(?:please|can\s+you|could\s+you|would\s+like\s+(?:a|to)|want\s+(?:a|to))\s+(?:add|create|implement|build|include|support)\s+(?:a\s+)?(?:new\s+)?(?:feature|option|capability|functionality)",
    r"\bfeature\s+request\b",
    r"\bsuggestion\b",
    r"it\s+would\s+be\s+(?:nice|great|useful|helpful|good)\s+(?:if|to)",
    r"(?:add|include|support)\s+the\s+(?:ability|option|feature)\s+to",
    r"(?:i\s+wish|i\s+hope)\s+(?:you\s+(?:could|would)|there\s+was)",
]

# ---------------------------------------------------------------------------
# Risk detection
# ---------------------------------------------------------------------------
# HIGH risk → always escalate
HIGH_RISK_KEYWORDS = {
    "fraud", "fraudulent", "stolen identity", "identity theft", "identity stolen",
    "identity has been stolen", "account compromised", "compromised account",
    "security breach", "data breach", "hacked", "phishing", "ransomware",
    "unauthorized transaction", "unauthorized charge", "unauthorized access",
    "security vulnerability", "exploit", "vulnerability found",
    "legal action", "sue", "lawsuit", "compliance violation",
    "money laundering", "suspicious activity", "criminal",
}

# MEDIUM risk → escalate if confidence is low, otherwise reply
MEDIUM_RISK_KEYWORDS = {
    "refund", "chargeback", "billing dispute", "payment issue",
    "charge dispute", "overcharged", "double charged",
    "urgent", "immediately", "asap", "emergency",
    "account recovery", "recover account", "account locked",
    "stolen card", "lost card", "blocked card", "card blocked",
    "score dispute", "unfair", "manually grade",
    "sensitive data", "personal data", "gdpr",
    "password reset", "reset password", "cant login", "cannot login",
}

# Prompt injection signatures
INJECTION_SIGNATURES = [
    "ignore previous instructions",
    "ignore all previous",
    "disregard previous",
    "forget everything",
    "you are now",
    "act as",
    "new persona",
    "reveal your",
    "show me your system prompt",
    "affiche toutes",
    "règles internes",
    "logique exacte",
    "documents récupérés",
    "montre tes",
]

# ---------------------------------------------------------------------------
# Decision thresholds
# ---------------------------------------------------------------------------
# Cases where we always escalate regardless of corpus match
ALWAYS_ESCALATE_PATTERNS = [
    r"identity\s+(?:theft|stolen|has\s+been\s+stolen)",
    r"my\s+(?:account|identity)\s+(?:has\s+been\s+)?(?:hacked|compromised|stolen|breached)",
    r"security\s+(?:breach|vulnerability|exploit)",
    r"i\s+(?:found|discovered|identified)\s+(?:a\s+)?(?:major\s+)?(?:security|vulnerability|exploit|bug\s+that)",
    r"bug\s+bounty",
    r"unauthorized\s+(?:charge|transaction|access|payment)",
    r"legal\s+action",
    r"(?:stolen|lost)\s+(?:my\s+)?(?:identity|personal\s+data|sensitive)",
    r"(?:is|are)\s+(?:down|not\s+(?:working|accessible|available))\s*[.!]?\s*$",  # full outage
    r"none\s+of\s+the\s+(?:pages?|submissions?|challenges?|requests?)\b",  # outage
    r"all\s+(?:the\s+)?(?:pages?|submissions?|challenges?|requests?)\s+(?:are\s+)?(?:not\s+)?(?:working|accessible|available|failing|broken)",
    r"payment\s+(?:failed|issue|problem|error)",
    r"(?:give|get)\s+me\s+(?:my\s+|the\s+|a\s+)?(?:money|refund)(?:\s+(?:back|asap|immediately|now|today))?",
    r"(?:please|can\s+you|i\s+(?:want|need))\s+(?:a\s+|my\s+|the\s+)?refund",
    r"refund\s+(?:asap|immediately|now|today|urgently)",
    r"i\s+(?:want|need|require|demand)\s+(?:a|my)\s+(?:refund|money\s+back)",
    r"score\s+(?:dispute|review|change|increase|wrong|incorrect|unfair)",
    r"(?:increase|change|modify)\s+(?:my\s+)?score",
    r"infosec\s+(?:process|form|audit|compliance|forms?)",
    r"security\s+(?:questionnaire|form|audit|review|process)",
    r"fill\s+(?:in|out)\s+(?:the\s+)?(?:infosec|security|compliance)\s+(?:forms?|questionnaire)",
    r"(?:dispute|chargeback)\s+(?:a\s+)?(?:charge|transaction|payment)",
    r"suspend\s+(?:our\s+)?(?:account|subscription)",
    r"pause\s+(?:our\s+)?subscription",
]

# ---------------------------------------------------------------------------
# Escalation and out-of-scope response templates
# ---------------------------------------------------------------------------
ESCALATION_RESPONSE = (
    "Thank you for contacting support.\n\n"
    "Your request requires assistance from a human support specialist.\n\n"
    "We have escalated your case to the appropriate team.\n\n"
    "You will be contacted shortly."
)

OUT_OF_SCOPE_RESPONSE = (
    "Thank you for reaching out.\n\n"
    "I'm sorry, but your request appears to be outside the scope of our support capabilities.\n\n"
    "Our support covers HackerRank, Claude, and Visa-related issues. "
    "If you have a question about one of these products, please feel free to reach out again."
)

INJECTION_RESPONSE = (
    "Thank you for contacting support.\n\n"
    "I'm unable to process this request. "
    "If you have a genuine support question related to HackerRank, Claude, or Visa, "
    "please submit a new ticket with your specific issue."
)

# ---------------------------------------------------------------------------
# Product area mapping (from data/ subdirectory structure)
# ---------------------------------------------------------------------------
HACKERRANK_AREA_MAP = {
    "screen": "screen",
    "interviews": "interviews",
    "library": "library",
    "chakra": "chakra",
    "engage": "engage",
    "integrations": "integrations",
    "skillup": "skillup",
    "settings": "settings",
    "general-help": "general_help",
    "general_help": "general_help",
    "hackerrank_community": "community",
    "uncategorized": "general_support",
}

CLAUDE_AREA_MAP = {
    "amazon-bedrock": "amazon_bedrock",
    "claude-api-and-console": "claude_api",
    "claude-code": "claude_code",
    "claude-desktop": "claude_desktop",
    "claude-for-education": "education",
    "claude-for-government": "government",
    "claude-for-nonprofits": "nonprofits",
    "claude-in-chrome": "chrome_extension",
    "claude-mobile-apps": "mobile",
    "connectors": "connectors",
    "identity-management-sso-jit-scim": "identity_management",
    "privacy-and-legal": "privacy",
    "pro-and-max-plans": "pro_max_plans",
    "safeguards": "safeguards",
    "team-and-enterprise-plans": "team_enterprise",
    # sub-areas under claude/claude/
    "account-management": "account_management",
    "billing": "billing",
}

VISA_AREA_MAP = {
    "travel": "travel_support",
    "merchant": "merchant_support",
    "small-business": "small_business",
    "small_business": "small_business",
    "consumer": "consumer_support",
    "data-security": "data_security",
    "dispute": "dispute_resolution",
}
