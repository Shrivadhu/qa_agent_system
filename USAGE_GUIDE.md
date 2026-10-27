# QA Agent System - Usage Guide

## Table of Contents
1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Database Setup](#database-setup)
4. [Running the API Server](#running-the-api-server)
5. [Analyzing Sessions](#analyzing-sessions)
6. [API Endpoints](#api-endpoints)
7. [Understanding Results](#understanding-results)
8. [Testing](#testing)
9. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites
- Python 3.9 or higher (3.11 used in Docker)
- PostgreSQL 12 or higher
- pip package manager

### Step 1: Clone or Extract the Project

```bash
cd qa_agent_system
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### Step 4: Download NLTK Data

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Database Configuration
DB_PASSWORD=your_secure_password

# Optional: Override other settings
LOG_LEVEL=INFO
API_PORT=8000
```

### Configuration File

Edit `config/config.yaml` to customize:

```yaml
database:
  host: localhost
  port: 5432
  name: qa_agent_db
  user: qa_agent_user
  password: ${DB_PASSWORD}

models:
  bert_base: "bert-base-uncased"
  sentiment_model: "distilbert-base-uncased-finetuned-sst-2-english"

verdict:
  thresholds:
    successful: 0.90
    partial: 0.70
    dispute: 0.50
```

## Database Setup

### Create Database

```sql
-- Connect to PostgreSQL
psql -U postgres

-- Create database and user
CREATE DATABASE qa_agent_db;
CREATE USER qa_agent_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE qa_agent_db TO qa_agent_user;
```

### Initialize Tables

```bash
python main.py init-db
```

This creates all required tables:
- `learning_outcomes`
- `qa_reports`
- `keyword_analyses`
- `feedback_authenticity`

## Running the API Server

### Start the Server

```bash
python main.py server
```

The server will start on `http://localhost:8000`

### API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Analyzing Sessions

### Method 1: Using the CLI

```bash
python main.py analyze example_session.json
```

### Method 2: Using the API

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d @example_session.json
```

### Method 3: Using Python Code

```python
from api.qa_service import QAService
from models.schemas import SessionAnalysisRequest

# Initialize service
qa_service = QAService()

# Prepare request
request = SessionAnalysisRequest(
    session_id="session_123",
    transcript="...",
    promised_skills=["python_basics"],
    feedback_data={"free_text_comment": "..."},
    participant_ratings={"clarity": 5.0}
)

# Analyze
result = qa_service.analyze_session(request)

print(f"Verdict: {result.qa_verdict}")
print(f"Score: {result.overall_quality_score}")
```

## API Endpoints

### Security
If `API_KEY` is set in the environment or `.env`, requests to `/analyze`, `/skills`, and `/report/{session_id}` must include the header:
```
X-API-Key: your_secret_api_key
```
`/health` and `/` remain open without authentication for health checking.

### GET /skills

Retrieve supported skills and their expected taxonomy keywords.

**Headers:**
`X-API-Key: <key>` (if authentication enabled)

**Response:**
```json
{
  "skills": ["python_basics", "python_functions", "data_structures", ...],
  "taxonomy": {
    "python_basics": ["variable", "function", "loop", ...],
    ...
  }
}
```

### POST /analyze

Analyze a teaching session and generate QA report.

**Request Body:**
```json
{
  "session_id": "string",
  "transcript": "string",
  "promised_skills": ["python_basics"],
  "feedback_data": {
    "free_text_comment": "string"
  },
  "participant_ratings": {
    "clarity": 5.0,
    "helpfulness": 5.0
  },
  "skill_keywords": {
    "custom_skill": ["keyword1", "keyword2"]
  }
}
```

**Response:**
```json
{
  "qa_report_id": 1,
  "session_id": "session_123",
  "qa_verdict": "successful",
  "overall_quality_score": 0.89,
  "component_scores": {
    "scope_completeness": 0.85,
    "teaching_quality": 0.90,
    "learning_indicators": 0.88,
    "feedback_authenticity": 0.92,
    "participant_satisfaction": 0.95,
    "compliance": 1.0
  },
  "evidence": "Session received a SUCCESSFUL verdict...",
  "recommendation": "settle",
  "created_at": "2025-01-29T10:30:00Z"
}
```

### GET /report/{session_id}

Retrieve existing QA report.

**Response:** Same as POST /analyze

### GET /health

Health check endpoint that verifies server and database connectivity.

**Response:**
```json
{
  "status": "healthy",
  "database": "connected"
}
```

### POST /analyze/batch

Analyze multiple sessions in a single batch request with partial-failure tolerance.

**Request Body:**
```json
{
  "sessions": [
    {
      "session_id": "session_001",
      "transcript": "...",
      "promised_skills": ["python_basics"],
      "feedback_data": {"free_text_comment": "..."}
    },
    {
      "session_id": "session_002",
      "transcript": "...",
      "promised_skills": ["web_development"],
      "feedback_data": {"free_text_comment": "..."}
    }
  ]
}
```

**Response:**
```json
{
  "total": 2,
  "successful": 2,
  "failed": 0,
  "results": [ ... ],
  "errors": [ ]
}
```

### GET /reports

Retrieve saved QA reports with optional filters and pagination.

**Query Parameters:**
- `verdict` (optional): Filter by `successful`, `partial`, `dispute`, `failed`
- `recommendation` (optional): Filter by `settle`, `review`, `escalate`
- `limit` (default: 50): Number of records
- `offset` (default: 0): Pagination offset

### GET /reports/summary

Retrieve system-wide aggregated QA quality metrics and score distributions.

**Response:**
```json
{
  "total_sessions_analyzed": 42,
  "verdict_distribution": {
    "successful": 28,
    "partial": 10,
    "dispute": 3,
    "failed": 1
  },
  "recommendation_distribution": {
    "settle": 28,
    "review": 10,
    "escalate": 4
  },
  "average_scores": {
    "overall_quality": 0.842,
    "scope_completeness": 0.795,
    "teaching_quality": 0.865,
    "learning_indicators": 0.880,
    "feedback_authenticity": 0.912,
    "participant_satisfaction": 0.940,
    "compliance": 0.985
  }
}
```

## Understanding Results

### QA Verdicts

- **SUCCESSFUL** (≥0.90): Session met all quality standards
- **PARTIAL** (0.70-0.89): Session was good but had minor issues
- **DISPUTE** (0.50-0.69): Significant quality concerns
- **FAILED** (<0.50): Session did not meet minimum standards

### Component Scores

All scores range from 0.0 to 1.0:

1. **Scope Completeness**: Coverage of promised topics
2. **Teaching Quality**: Clarity, engagement, patience, structure, relevance
3. **Learning Indicators**: Evidence of learner progression
4. **Feedback Authenticity**: Genuineness of feedback
5. **Participant Satisfaction**: Overall learner satisfaction
6. **Compliance**: Adherence to policies

### Recommendations

- **SETTLE**: Automatically approve and settle payment
- **REVIEW**: Manual review recommended
- **ESCALATE**: Serious issues require immediate attention

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test

```bash
pytest tests/test_qa_system.py::test_keyword_extraction -v
```

### Run with Coverage

```bash
pytest tests/ --cov=. --cov-report=html
```

View coverage report: `htmlcov/index.html`

## Troubleshooting

### Issue: Database Connection Error

**Solution:** Verify PostgreSQL is running and credentials are correct
```bash
psql -U qa_agent_user -d qa_agent_db -h localhost
```

### Issue: Model Download Errors

**Solution:** Models download on first use. Ensure internet connection:
```python
from transformers import AutoModel
AutoModel.from_pretrained("bert-base-uncased")
```

### Issue: Out of Memory

**Solution:** Reduce batch size in config:
```yaml
models:
  batch_size: 16  # Reduce from 32
```

### Issue: Slow Analysis

**Optimization tips:**
1. Enable caching in config
2. Use GPU if available
3. Reduce max_keywords in keyword extraction
4. Use smaller models (distilbert instead of bert)

## Production Deployment

### Using Gunicorn

```bash
gunicorn api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
```

### Using Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py", "server"]
```

### Environment Variables for Production

```bash
export DB_PASSWORD=secure_production_password
export LOG_LEVEL=WARNING
export API_WORKERS=8
```

## Support

For issues or questions:
1. Check the logs in `logs/qa_agent.log`
2. Review the API documentation at `/docs`
3. Run tests to verify installation
4. Check configuration in `config/config.yaml`
