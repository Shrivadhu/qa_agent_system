# QA Agent Post-Session Analysis System

## Overview
Automated Quality Assurance system for analyzing post-session tutoring and teaching transcripts. It evaluates objective coverage against promised skills, scores teaching quality across pedagogical dimensions, tracks learner knowledge progression, validates feedback authenticity and sentiment, performs policy compliance checking, and calculates actionable QA verdicts.

---

## Key Features & Architecture

### 6.1 Objective Verification
- **Keyword Extraction**: NLP keyword extraction using TF-IDF and YAKE, returning top keywords and frequencies with stop-word filtering.
- **Scope Mapping & Taxonomy**: Configurable skill taxonomy in `config/skill_taxonomy.json` supporting 9 core domains, plus custom request-level `skill_keywords`.
- **Coverage Checking**: Compares extracted session topics with expected skills and identifies covered vs. missing topics.

### 6.2 Teaching Quality Assessment
- **Pedagogical Classification**: Scores transcript segments on 5 dimensions: Clarity, Engagement, Patience, Structure, and Relevance (0.0–1.0).
- **Quality Aggregation**: Weighted average across transcript segments weighted by segment length and dimension weights.

### 6.3 Learning Outcome Verification
- **Knowledge Progression Tracking**: Detects learner questions, statements of understanding, and demonstration of knowledge.
- **Learning Outcome Persistence**: Generates a progression graph over time and stores progression metrics in the database.

### 6.4 Feedback Validation
- **Sentiment & Tone Consistency**: Analyzes feedback sentiment and validates consistency against participant ratings.
- **Plagiarism Detection**: Stores past session feedback comments in the database and detects copied or duplicate feedback using embedding similarity (> 0.90 threshold).
- **Template Detection & Human-Likeness**: Flags formulaic template matches and evaluates natural sentence variance, imperfections, and specific details.
- **Transcript Cross-Validation**: Validates feedback claims directly against transcript evidence.

### 6.5 Policy Compliance Checking
- **Policy Enforcement**: Flags short transcripts (< 50 words), shared contact details (emails, phone numbers), external messaging mentions (WhatsApp, Telegram), off-platform payment talk (PayPal, Venmo, Crypto, Wire), and abusive/inappropriate language.
- **Evidence Inclusion**: Detailed violations are appended directly to the QA report's evidence summary.

### 6.6 QA Verdict & Reporting
- **Equal 6-Component Aggregation**: Scope completeness, Teaching quality, Learning indicators, Feedback authenticity, Participant satisfaction, and Compliance (1/6 each).
- **Verdict Thresholds**:
  - `>= 0.90`: `SUCCESSFUL` (Recommendation: `SETTLE`)
  - `0.70 – 0.89`: `PARTIAL` (Recommendation: `REVIEW`)
  - `0.50 – 0.69`: `DISPUTE` (Recommendation: `REVIEW` or `ESCALATE`)
  - `< 0.50`: `FAILED` (Recommendation: `ESCALATE`)

---

## Project Structure

```
qa_agent_system/
├── Dockerfile                  # Non-root container image with health check
├── docker-compose.yml          # Postgres + Redis + QA API compose stack
├── render.yaml                 # Render cloud blueprint
├── DEPLOY.md                   # Complete production deployment guide
├── USAGE_GUIDE.md              # Detailed API and CLI usage manual
├── requirements.txt            # Python dependencies
├── main.py                     # CLI entry point (server, analyze, init-db)
├── config/
│   ├── __init__.py             # Dotenv & configuration manager
│   ├── config.yaml             # Core weights, thresholds, and parameters
│   └── skill_taxonomy.json     # Editable skill keywords taxonomy
├── models/
│   ├── __init__.py
│   └── schemas.py              # SQLAlchemy models & Pydantic schemas
├── core/
│   ├── objective_verification/ # Keyword extractor & coverage checker
│   ├── teaching_quality/       # Quality classifier & aggregator
│   ├── learning_outcomes/      # Progression tracker
│   ├── feedback_validation/    # Sentiment, authenticity, cross-validator
│   ├── compliance/             # Policy and compliance checker
│   └── verdict/                # QA score aggregator & verdict calculator
├── database/
│   ├── __init__.py
│   └── repository.py           # Auto table creation & database operations
├── api/
│   ├── __init__.py             # FastAPI app, API key auth, CORS, /health, /skills
│   └── qa_service.py           # Master orchestration service
├── utils/
│   ├── embeddings.py           # Neural embeddings with TF-IDF fallback
│   ├── nlp_utils.py            # Sentence & word tokenization, YAKE, TF-IDF
│   └── logger.py               # Loguru structured logging
└── tests/
    ├── conftest.py
    └── test_qa_system.py       # 18 unit and integration tests
```

---

## Quick Start

### 1. Run Unit Tests (18 tests passing)
```bash
python -m pytest tests/ -v
```

### 2. Analyze a Session via CLI
```bash
export DATABASE_URL="sqlite:///./dev.db"
python main.py analyze example_session.json
```

### 3. Start the API Server
```bash
python main.py server
```
Server runs at `http://localhost:8000`. Interactive docs are available at `http://localhost:8000/docs`.

---

## Production Deployment

Detailed deployment instructions for Docker, Render, and VPS are available in [DEPLOY.md](DEPLOY.md).

### Docker Compose
```bash
cp .env.example .env
# Set DB_PASSWORD and API_KEY in .env
docker compose up -d --build
curl http://localhost:8000/health
```

---

## Security & Authentication

Set `API_KEY=your_secret_token` in `.env` or system environment.
Clients must provide the key via the `X-API-Key` header:
```bash
curl -H "X-API-Key: your_secret_token" http://localhost:8000/skills
```
The `/health` and `/` endpoints remain open for load balancer and uptime checks.
