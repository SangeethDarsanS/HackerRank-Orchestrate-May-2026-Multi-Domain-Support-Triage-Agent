# Multi-Domain Support Triage Agent

A production-ready, terminal-based AI support triage system that resolves support tickets across three ecosystems: **HackerRank**, **Claude (Anthropic)**, and **Visa**.

The agent works **fully offline** using only the provided support corpus under `data/`.  
No external APIs or LLMs are called at inference time.

---

## Architecture

```
main.py
│
├── modules/loader.py          — parse & chunk 773 markdown docs from data/
├── modules/indexer.py         — sentence-transformer embeddings + FAISS index
├── modules/retriever.py       — semantic search + domain filter + keyword rerank
├── modules/classifier.py      — domain / request_type / product_area classifier
├── modules/risk_engine.py     — risk level + injection + out-of-scope detection
├── modules/decision_engine.py — reply vs escalate decision logic
├── modules/response_generator.py — extractive grounded response from corpus
├── modules/logger.py          — structured ticket log + AGENTS.md session log
└── modules/evaluator.py       — accuracy / F1 / escalation rate metrics
```

### Pipeline

```
Ticket Input (CSV)
        ↓
   Preprocess (normalise columns)
        ↓
   Request Type Classification  (keyword-based, deterministic)
        ↓
   Domain Classification        (Company field → keywords → retrieval)
        ↓
   Risk Detection               (keyword rules + injection detection)
        ↓
   Vector Retrieval             (FAISS cosine search over chunked corpus)
        ↓
   Product Area Resolution      (from retrieval + keyword fallback)
        ↓
   Decision (replied / escalated)
        ↓
   Response Generation          (extractive from top corpus chunks)
        ↓
   Output CSV + Log
```

---

## Quick Start

### 1. Install dependencies

```bash
cd code/
pip install -r requirements.txt
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB).  
Subsequent runs load it from the local Hugging Face cache.

### 2. Run the agent

```bash
python main.py --tickets ../support_tickets/support_tickets.csv
```

Outputs:

| File | Description |
|------|-------------|
| `../support_tickets/output.csv` | Predictions (one row per ticket) |
| `../support_tickets/log.txt` | Structured per-ticket operation log |

### 3. Evaluate (backtesting)

```bash
python evaluate.py
```

Compares `output.csv` against `sample_support_tickets.csv` and prints accuracy,
precision, recall, F1, escalation rate, and latency.

### 4. Docker

```bash
# From repo root
docker build -f code/Dockerfile -t triage-agent .
docker run --rm -v "$(pwd)/support_tickets:/app/support_tickets" triage-agent
```

---

## CLI Reference

```
python main.py [OPTIONS]

Options:
  --tickets PATH          Input CSV  (default: ../support_tickets/support_tickets.csv)
  --output  PATH          Output CSV (default: ../support_tickets/output.csv)
  --log     PATH          Log file   (default: ../support_tickets/log.txt)
  --rebuild-index         Force rebuild the FAISS index (ignore cache)
  --top-k   INT           Chunks retrieved per query    (default: 10)
```

---

## Output Schema

`output.csv` columns:

| Column | Values | Description |
|--------|--------|-------------|
| `issue` | text | Original issue text |
| `subject` | text | Original subject |
| `company` | text | Original company |
| `response` | text | User-facing reply or escalation message |
| `product_area` | string | Support category (e.g. `screen`, `billing`, `travel_support`) |
| `status` | `replied` / `escalated` | Triage decision |
| `request_type` | `product_issue` / `feature_request` / `bug` / `invalid` | Request classification |
| `justification` | text | Reasoning trace for the decision |

---

## Design Decisions

### Why extractive (not generative) responses?
The challenge requires responses grounded **exclusively** in the provided corpus.  
Extractive retrieval guarantees zero hallucination — every sentence in the response
comes verbatim from the indexed documents.

### Why FAISS + sentence-transformers?
`all-MiniLM-L6-v2` provides strong semantic similarity at ~384-dim, fitting well
within the memory budget.  FAISS flat-IP over normalised vectors gives exact
cosine similarity in O(n) — fast enough for < 1s per ticket.

### Why keyword re-ranking on top of semantic search?
Semantic search alone can surface thematically similar but technically wrong docs.
Keyword overlap re-ranking ensures that specific terms (e.g. "zoom connectivity",
"inactivity time") boost the most on-point documents.

### Escalation logic
- **Always escalate**: fraud, identity theft, security breaches, billing disputes,
  score disputes, prompt injections, site-wide outages.
- **Escalate on low confidence**: medium-risk tickets where no corpus match exceeds
  the similarity threshold.
- **Reply**: everything else, using corpus-grounded extractive content.

### Safety
- Prompt injection detection (English + French patterns).
- No internal reasoning ever exposed in responses.
- No policy claims outside the corpus.

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Latency per ticket | < 1 s | After index is loaded |
| Memory usage | < 1 GB | FAISS flat index + ST model |
| Deterministic | Yes | Seeded; no stochastic LLM calls |

---

## File Structure

```
.
├── AGENTS.md
├── README.md
├── data/
│   ├── claude/
│   ├── hackerrank/
│   └── visa/
├── support_tickets/
│   ├── support_tickets.csv      ← input
│   ├── sample_support_tickets.csv
│   ├── output.csv               ← generated
│   └── log.txt                  ← generated
└── code/
    ├── main.py
    ├── evaluate.py
    ├── requirements.txt
    ├── Dockerfile
    ├── run.sh
    ├── README.md                ← this file
    └── modules/
        ├── config.py
        ├── loader.py
        ├── indexer.py
        ├── retriever.py
        ├── classifier.py
        ├── risk_engine.py
        ├── decision_engine.py
        ├── response_generator.py
        ├── logger.py
        └── evaluator.py
```

---

## Chat Transcript Logging

Per AGENTS.md §2, the development chat transcript is appended to:

- **Windows**: `%USERPROFILE%\hackerrank_orchestrate\log.txt`
- **Linux/macOS**: `$HOME/hackerrank_orchestrate/log.txt`

The triage operation log (per-ticket details) is at:

```
support_tickets/log.txt
```

---

## Environment Variables

No external API keys are required (fully offline).  
If you extend the agent with an LLM, add your key to `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

The `.env` file is gitignored.
