"""
Multi-aspect classifier:
  - domain        (hackerrank | claude | visa | unknown)
  - request_type  (product_issue | feature_request | bug | invalid)
  - product_area  (derived from retrieval results + rule-based fallback)
"""

import logging
import pickle
import re
from pathlib import Path
from typing import Optional, Tuple

from .config import (
    DOMAINS, DOMAIN_KEYWORDS,
    INVALID_PATTERNS, BUG_PATTERNS, FEATURE_REQUEST_PATTERNS,
    INJECTION_SIGNATURES,
)
from .retriever import RetrievalResult

logger = logging.getLogger(__name__)

# Path for caching the trained product-area model
_CODE_DIR        = Path(__file__).resolve().parent.parent   # …/code/
_PROJECT_DIR     = _CODE_DIR.parent                          # …/hackerrank-orchestrate-may26/
_PA_MODEL_CACHE  = _CODE_DIR / ".cache" / "product_area_model.pkl"
_PA_TRAINING_CSV = _PROJECT_DIR / "data" / "product_area_training.csv"
_PA_ML_THRESHOLD = 0.60   # minimum ML confidence to trust the prediction


# ---------------------------------------------------------------------------
# Domain classifier
# ---------------------------------------------------------------------------

class DomainClassifier:
    """
    Determines the product domain for a ticket.

    Priority:
      1. Company field (if present and valid)
      2. Keyword matching against issue + subject text
      3. Retrieval-based domain (from best-matching corpus chunk)
    """

    _COMPANY_MAP = {
        "hackerrank": "hackerrank",
        "hacker rank": "hackerrank",
        "claude": "claude",
        "anthropic": "claude",
        "visa": "visa",
        "none": None,
        "": None,
    }

    def classify(
        self,
        company: str,
        text: str,
        retrieval: Optional[RetrievalResult] = None,
    ) -> str:
        # 1. Company field
        domain = self._from_company(company)
        if domain:
            return domain

        # 2. Keyword matching
        domain = self._from_keywords(text)
        if domain:
            return domain

        # 3. Retrieval-based domain
        if retrieval and retrieval.chunks:
            return retrieval.best_domain

        return "unknown"

    def _from_company(self, company: str) -> Optional[str]:
        key = company.strip().lower()
        return self._COMPANY_MAP.get(key)

    def _from_keywords(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        scores: dict[str, int] = {d: 0 for d in DOMAINS}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            for kw in keywords:
                if re.search(r'(?<!\w)' + re.escape(kw) + r'(?!\w)', text_lower):
                    scores[domain] += 1
        best = max(scores, key=lambda d: scores[d])
        return best if scores[best] > 0 else None


# ---------------------------------------------------------------------------
# Request type classifier
# ---------------------------------------------------------------------------

class RequestTypeClassifier:
    """
    Maps ticket text → one of:
      invalid | bug | feature_request | product_issue
    """

    def classify(self, text: str) -> str:
        t = text.strip().lower()

        # 1. Prompt injection / clearly malicious → invalid
        if self._is_injection(text):
            return "invalid"

        # 2. Explicit invalid patterns (out-of-scope, greetings, etc.)
        for pattern in INVALID_PATTERNS:
            if re.search(pattern, t, re.IGNORECASE):
                return "invalid"

        # 3. Bug signals
        for pattern in BUG_PATTERNS:
            if re.search(pattern, t, re.IGNORECASE):
                return "bug"

        # 4. Feature request signals
        for pattern in FEATURE_REQUEST_PATTERNS:
            if re.search(pattern, t, re.IGNORECASE):
                return "feature_request"

        # 5. Default
        return "product_issue"

    def _is_injection(self, text: str) -> bool:
        t = text.lower()
        return any(sig in t for sig in INJECTION_SIGNATURES)


# ---------------------------------------------------------------------------
# Product area resolver
# ---------------------------------------------------------------------------

class ProductAreaResolver:
    """
    Determines the product_area label for a ticket from (priority order):

      1. Rule boosting  — explicit high-confidence keyword → area mappings
      2. ML classifier  — TF-IDF + LogReg trained on product_area_training.csv
                          (only used when confidence >= _PA_ML_THRESHOLD = 0.60;
                           otherwise falls back to domain classifier)
      3. Retrieval-based — area from the best-matching corpus chunk
      4. Keyword heuristic — word-boundary keyword → area count
      5. Domain-level default
    """

    # ---------------------------------------------------------------------------
    # High-priority rule boosts (keyword → area, with minimum confidence).
    # These fire before retrieval and ML, giving determinate outcomes for
    # unambiguous signals.
    # ---------------------------------------------------------------------------
    _RULE_BOOSTS: list = [
        # (keywords_list, area, confidence)
        # HackerRank
        (["mock interview", "skillup"],                               "skillup",           0.90),
        (["inactivity", "hr lobby", "codepair"],                     "interviews",         0.90),
        (["assessment", "test invitation", "test link", "proctoring","reschedule.*test",
          "reschedule.*assessment"],                                  "screen",             0.85),
        (["resume builder", "apply tab", "apply to job"],            "community",          0.85),
        (["remove user", "remove.*employee", "pause.*subscription",
          "user management", "infosec.*form", "hiring.*account"],    "settings",           0.85),
        (["bedrock", "aws bedrock", "amazon bedrock"],               "amazon_bedrock",     0.95),
        (["lti", "professor", "classroom integration"],              "education",          0.92),
        (["bug bounty", "vulnerability report", "security.*claude"], "safeguards",         0.90),
        (["crawl.*website", "crawling.*website", "website.*crawl"],  "privacy",            0.90),
        (["delete.*conversation", "conversation.*history.*delete",
          "personal.*data.*used", "data.*model.*train"],             "privacy",            0.85),
        (["traveller.*cheque", "travelers.*cheque", "cheque.*stolen",
          "cheque.*lost"],                                            "travel_support",     0.92),
        (["dispute.*charge", "chargeback", "wrong.*product.*visa",
          "merchant.*refund"],                                        "dispute_resolution", 0.90),
        (["identity.*theft", "card.*stolen", "stolen.*card",
          "lost.*card", "card.*blocked", "urgent.*cash.*visa",
          "minimum.*spend.*visa"],                                    "consumer_support",   0.88),
    ]

    # Keyword → area overrides (applied when retrieval area is weak)
    _AREA_KEYWORDS: dict[str, list] = {
        # HackerRank
        "screen": ["screen", "test", "assessment", "candidate", "invite"],
        "interviews": ["interview", "interviewer", "inactivity", "screen share", "hr lobby"],
        "library": ["question library", "question bank", "coding challenge"],
        "community": ["community", "profile", "resume", "apply", "practice"],
        "settings": ["setting", "admin", "user management", "remove user", "subscription", "seat"],
        "chakra": ["chakra", "ai interview", "ai interviewer"],
        "skillup": ["skillup", "skill up", "mock", "mock interview"],
        "integrations": ["integration", "ats", "greenhouse", "ashby", "webhook"],
        # Claude
        "account_management": ["account", "login", "logout", "session", "workspace", "seat", "access"],
        "billing": ["billing", "payment", "invoice", "charge", "subscription", "coupon", "plan"],
        "privacy": ["privacy", "data", "gdpr", "conversation history", "delete conversation", "crawl"],
        "safeguards": ["safety", "harm", "policy", "vulnerability", "security", "bug bounty"],
        "amazon_bedrock": ["bedrock", "aws", "amazon", "region"],
        "claude_api": ["api", "sdk", "console", "token", "rate limit"],
        "claude_code": ["claude code", "ide", "editor", "vscode"],
        "education": ["student", "professor", "lti", "university", "college", "classroom"],
        "team_enterprise": ["team", "enterprise", "workspace", "organization", "sso"],
        # Visa
        "travel_support": ["travel", "traveller", "cheque", "foreign", "abroad", "trip", "lisbon"],
        "consumer_support": ["consumer", "cardholder", "personal card", "visa card", "lost card", "stolen card", "blocked card"],
        "dispute_resolution": ["dispute", "chargeback", "wrong product", "refund", "merchant"],
        "small_business": ["small business", "merchant", "business account"],
        "data_security": ["data security", "pci", "compliance"],
    }

    _defaults = {
        "hackerrank": "general_help",
        "claude":     "claude",
        "visa":       "general_support",
        "unknown":    "general_support",
    }

    def __init__(self) -> None:
        self._ml_model = self._load_or_train_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        retrieval: Optional[RetrievalResult],
        domain: str,
        text: str,
    ) -> str:
        # --- Priority 1: Rule boosting ---
        area, _ = self._rule_boost(text)
        if area:
            return area

        # --- Priority 2: ML classifier ---
        if self._ml_model is not None:
            area, conf = self._ml_predict(text)
            if area and conf >= _PA_ML_THRESHOLD:
                return area
            # confidence < threshold → fall through to retrieval/keyword

        # --- Priority 3: Retrieval-based area ---
        if retrieval and retrieval.chunks:
            domain_matched = [
                (chunk, score)
                for chunk, score in retrieval.chunks
                if domain == "unknown" or chunk.domain == domain
            ]
            if domain_matched:
                best_chunk = max(domain_matched, key=lambda x: x[1])[0]
                area = best_chunk.area
                if area and area != "general_support":
                    return area

        # --- Priority 4: Keyword heuristic (word-boundary safe) ---
        text_lower = text.lower()
        best_area  = None
        best_count = 0
        for kw_area, keywords in self._AREA_KEYWORDS.items():
            count = sum(
                1 for kw in keywords
                if re.search(r'(?<!\w)' + re.escape(kw) + r'(?!\w)', text_lower)
            )
            if count > best_count:
                best_count = count
                best_area  = kw_area

        if best_area and best_count > 0:
            return best_area

        # --- Priority 5: Domain-level default ---
        return self._defaults.get(domain, "general_support")

    # ------------------------------------------------------------------
    # Rule boost
    # ------------------------------------------------------------------

    def _rule_boost(self, text: str) -> Tuple[Optional[str], float]:
        """
        Check high-priority keyword rules.

        Returns (area, confidence) if a rule fires, else (None, 0.0).
        Patterns support simple regex; matching is case-insensitive.
        """
        text_lower = text.lower()
        for keywords, area, conf in self._RULE_BOOSTS:
            for kw in keywords:
                if re.search(kw, text_lower):
                    return area, conf
        return None, 0.0

    # ------------------------------------------------------------------
    # ML classifier
    # ------------------------------------------------------------------

    def _ml_predict(self, text: str) -> Tuple[Optional[str], float]:
        """Run the TF-IDF + LogReg model and return (area, probability)."""
        try:
            vectoriser, clf = self._ml_model
            X = vectoriser.transform([text])
            proba = clf.predict_proba(X)[0]
            best_idx  = int(proba.argmax())
            best_conf = float(proba[best_idx])
            area = clf.classes_[best_idx]
            return area, best_conf
        except Exception as exc:
            logger.debug("[product_area_ml] predict failed: %s", exc)
            return None, 0.0

    def _load_or_train_model(self):
        """Load cached model or train a new one from product_area_training.csv."""
        # Try loading from cache first
        if _PA_MODEL_CACHE.exists():
            try:
                with open(_PA_MODEL_CACHE, "rb") as f:
                    model = pickle.load(f)
                logger.debug("[product_area_ml] loaded cached model from %s", _PA_MODEL_CACHE)
                return model
            except Exception as exc:
                logger.warning("[product_area_ml] cache load failed (%s) — retraining", exc)

        # Train from CSV
        if not _PA_TRAINING_CSV.exists():
            logger.info(
                "[product_area_ml] training data not found at %s — ML disabled",
                _PA_TRAINING_CSV,
            )
            return None

        try:
            import pandas as pd
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression

            df = pd.read_csv(_PA_TRAINING_CSV, dtype=str).fillna("")
            df.columns = [c.strip().lower() for c in df.columns]
            X_raw = df["text"].tolist()
            y     = df["product_area"].tolist()

            if len(set(y)) < 2:
                logger.warning("[product_area_ml] insufficient classes — ML disabled")
                return None

            vectoriser = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=6000,
                sublinear_tf=True,
                min_df=1,
            )
            X_vec = vectoriser.fit_transform(X_raw)

            clf = LogisticRegression(
                max_iter=400,
                C=2.0,
                random_state=42,
                solver="lbfgs",
            )
            clf.fit(X_vec, y)

            model = (vectoriser, clf)

            # Cache it
            try:
                _PA_MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
                with open(_PA_MODEL_CACHE, "wb") as f:
                    pickle.dump(model, f)
                logger.debug("[product_area_ml] model cached to %s", _PA_MODEL_CACHE)
            except Exception as exc:
                logger.warning("[product_area_ml] could not cache model: %s", exc)

            return model

        except Exception as exc:
            logger.warning("[product_area_ml] training failed: %s — ML disabled", exc)
            return None
