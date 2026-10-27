# QA Agent System Architecture

## System Overview

The QA Agent Post-Session Analysis System is a comprehensive platform for automated quality assurance of teaching sessions. It analyzes transcripts, evaluates teaching quality, validates feedback, and generates actionable verdicts.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  FastAPI Server (api/)                               │   │
│  │  - POST /analyze: Analyze session                    │   │
│  │  - GET /report/{id}: Retrieve report                 │   │
│  │  - GET /health: Health check                         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   QA Service Orchestrator                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  api/qa_service.py                                   │   │
│  │  Coordinates all analysis phases                     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Core Analysis Modules                     │
│                                                              │
│  ┌────────────────────┐  ┌─────────────────────┐           │
│  │ Phase 1: Objective │  │ Phase 2: Teaching   │           │
│  │ Verification       │  │ Quality             │           │
│  │ - KeywordExtractor │  │ - QualityClassifier │           │
│  │ - CoverageChecker  │  │ - QualityAggregator │           │
│  └────────────────────┘  └─────────────────────┘           │
│                                                              │
│  ┌────────────────────┐  ┌─────────────────────┐           │
│  │ Phase 3: Learning  │  │ Phase 4: Feedback   │           │
│  │ Outcomes           │  │ Validation          │           │
│  │ - ProgressionTrack │  │ - SentimentAnalyzer │           │
│  │                    │  │ - AuthenticityCheck │           │
│  │                    │  │ - CrossValidator    │           │
│  └────────────────────┘  └─────────────────────┘           │
│                                                              │
│  ┌────────────────────┐                                     │
│  │ Phase 5: Verdict   │                                     │
│  │ - VerdictCalc      │                                     │
│  └────────────────────┘                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     Utility Modules                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ NLP Utils    │  │ Embeddings   │  │ Logger       │      │
│  │ - Tokenize   │  │ - Encode     │  │ - Structured │      │
│  │ - Keywords   │  │ - Similarity │  │ - File/Stream│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Data Persistence Layer                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  PostgreSQL Database                                 │   │
│  │  - learning_outcomes                                 │   │
│  │  - qa_reports                                        │   │
│  │  - keyword_analyses                                  │   │
│  │  - feedback_authenticity                             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. API Layer (api/)

**Purpose:** REST API interface for system interaction

**Components:**
- `__init__.py`: FastAPI application and endpoints
- `qa_service.py`: Main service orchestrator

**Responsibilities:**
- Accept session analysis requests
- Return QA reports
- Handle errors and validation
- Provide health checks

### 2. Core Analysis Modules (core/)

#### 2.1 Objective Verification (core/objective_verification/)

**Purpose:** Verify that promised teaching objectives were met

**Components:**
- `keyword_extractor.py`: Extract keywords using TF-IDF and YAKE
- `coverage_checker.py`: Map keywords to skills and calculate coverage

**Algorithms:**
- TF-IDF vectorization for keyword importance
- YAKE for keyphrase extraction
- Semantic similarity using sentence transformers
- Coverage percentage calculation

**Output:** Scope completeness score (0.0-1.0)

#### 2.2 Teaching Quality (core/teaching_quality/)

**Purpose:** Assess quality of teaching across multiple dimensions

**Components:**
- `quality_classifier.py`: Classify teaching quality per segment
- `quality_aggregator.py`: Aggregate scores across transcript

**Dimensions Evaluated:**
- Clarity: Are explanations clear?
- Engagement: Is learner engaged?
- Patience: How patient is teacher?
- Structure: Is lesson organized?
- Relevance: Is teaching at appropriate level?

**Methods:**
- Rule-based classification (current)
- BERT-based classification (production-ready)
- Weighted aggregation

**Output:** Teaching quality score (0.0-1.0)

#### 2.3 Learning Outcomes (core/learning_outcomes/)

**Purpose:** Verify actual learning occurred

**Components:**
- `progression_tracker.py`: Track knowledge progression

**Detection Methods:**
- Question counting (confusion indicators)
- Understanding statement detection
- Progression trend analysis
- Engagement pattern recognition

**Output:** Learning indicators score (0.0-1.0)

#### 2.4 Feedback Validation (core/feedback_validation/)

**Purpose:** Validate authenticity and accuracy of feedback

**Components:**
- `sentiment_analyzer.py`: Analyze sentiment and consistency
- `authenticity_checker.py`: Detect plagiarism, templates, and bot feedback
- `cross_validator.py`: Validate claims against transcript

**Checks Performed:**
- Sentiment analysis (positive/neutral/negative)
- Sentiment-rating consistency
- Plagiarism detection (cosine similarity)
- Template matching
- Human-likeness scoring
- Evidence-based validation

**Output:** Feedback authenticity score (0.0-1.0)

#### 2.5 Verdict Calculation (core/verdict/)

**Purpose:** Calculate final verdict and recommendation

**Components:**
- `verdict_calculator.py`: Aggregate scores and determine verdict

**Verdict Thresholds:**
- ≥0.90: SUCCESSFUL
- 0.70-0.89: PARTIAL
- 0.50-0.69: DISPUTE
- <0.50: FAILED

**Recommendations:**
- SETTLE: Automatic approval
- REVIEW: Manual review needed
- ESCALATE: Immediate attention required

### 3. Utility Modules (utils/)

#### 3.1 NLP Utils (utils/nlp_utils.py)

**Purpose:** Common NLP operations

**Functions:**
- Text cleaning and preprocessing
- Tokenization (sentences, words)
- Keyword extraction (TF-IDF, YAKE)
- Text similarity calculation
- Pattern detection

#### 3.2 Embeddings (utils/embeddings.py)

**Purpose:** Text embedding and similarity

**Capabilities:**
- Text-to-vector encoding
- Cosine similarity calculation
- Semantic search
- Caching for performance

**Model:** Sentence-BERT (all-MiniLM-L6-v2)

#### 3.3 Logger (utils/logger.py)

**Purpose:** Structured logging

**Features:**
- Console and file logging
- Configurable log levels
- Rotation and retention
- Contextual information

### 4. Data Models (models/)

**Purpose:** Define data structures and schemas

**Components:**
- `schemas.py`: SQLAlchemy models and Pydantic schemas

**Models:**
- `LearningOutcome`: Learning progression data
- `QAReport`: Final QA analysis report
- `KeywordAnalysis`: Keyword extraction cache
- `FeedbackAuthenticity`: Feedback validation results

### 5. Database Layer (database/)

**Purpose:** Data persistence and retrieval

**Components:**
- `repository.py`: Database operations

**Database:** PostgreSQL

**Operations:**
- CRUD for all models
- Transaction management
- Connection pooling

### 6. Configuration (config/)

**Purpose:** System configuration management

**Components:**
- `config.yaml`: Configuration file
- `__init__.py`: Configuration loader

**Settings:**
- Database connection
- Model parameters
- Verdict thresholds
- Logging configuration
- API settings

## Data Flow

### Session Analysis Flow

```
1. Request Received (API)
   ↓
2. Initialize QA Service
   ↓
3. Phase 1: Objective Verification
   - Extract keywords from transcript
   - Generate expected keywords from skills
   - Calculate coverage
   - Score: scope_completeness
   ↓
4. Phase 2: Teaching Quality
   - Split transcript into segments
   - Classify each segment
   - Aggregate scores
   - Score: teaching_quality
   ↓
5. Phase 3: Learning Outcomes
   - Detect progression indicators
   - Track knowledge over time
   - Calculate progression score
   - Score: learning_indicators
   ↓
6. Phase 4: Feedback Validation
   - Analyze sentiment
   - Check authenticity (plagiarism, templates)
   - Validate against transcript
   - Scores: feedback_authenticity, participant_satisfaction
   ↓
7. Phase 5: Verdict Calculation
   - Aggregate all component scores
   - Apply weights
   - Determine verdict
   - Generate recommendation
   ↓
8. Persist Results (Database)
   - Save learning outcomes
   - Save QA report
   - Save analyses
   ↓
9. Return Response (API)
```

## Scalability Considerations

### Current Design

- **Single-threaded analysis**: Sequential processing of all phases
- **In-memory processing**: No distributed computing
- **Direct database access**: Synchronous operations

### Scaling Strategies

#### Horizontal Scaling

1. **Load Balancer**: Distribute requests across multiple API instances
2. **Database Read Replicas**: Scale read operations
3. **Redis Cache**: Cache frequently accessed data

#### Vertical Scaling

1. **GPU Acceleration**: Use GPU for ML models
2. **Increased Memory**: Handle larger transcripts
3. **Connection Pooling**: More concurrent database connections

#### Asynchronous Processing

1. **Celery Task Queue**: Offload analysis to background workers
2. **Message Broker**: Use RabbitMQ or Redis for task distribution
3. **Result Backend**: Store interim results in Redis

#### Microservices Architecture

Split into independent services:
- Keyword Extraction Service
- Teaching Quality Service
- Feedback Validation Service
- Verdict Calculation Service

## Security Considerations

### Authentication & Authorization

- API key authentication
- JWT tokens for session management
- Role-based access control

### Data Privacy

- Encrypt sensitive data at rest
- Secure database connections (SSL/TLS)
- PII data anonymization

### Input Validation

- Request payload validation (Pydantic)
- SQL injection prevention (SQLAlchemy ORM)
- XSS protection (FastAPI)

### Rate Limiting

- Per-IP rate limits
- Per-user rate limits
- Abuse detection

## Monitoring & Observability

### Metrics

- Request latency (p50, p95, p99)
- Error rates
- Database query performance
- Model inference time

### Logging

- Structured logging (JSON)
- Log aggregation (ELK stack)
- Error tracking (Sentry)

### Health Checks

- API health endpoint
- Database connectivity
- Model availability

## Performance Benchmarks

### Expected Performance

- **Small Session** (<500 words): ~5 seconds
- **Medium Session** (500-2000 words): ~10 seconds
- **Large Session** (>2000 words): ~20 seconds

### Bottlenecks

1. **ML Model Inference**: 40-50% of time
2. **Embedding Generation**: 20-30% of time
3. **Database Operations**: 10-15% of time
4. **Text Processing**: 10-15% of time

### Optimization Opportunities

1. Batch embedding generation
2. Model quantization
3. Caching strategies
4. Parallel phase execution

## Deployment Architecture

### Development

```
Developer Machine
├── Python 3.9+
├── PostgreSQL (local)
└── Virtual Environment
```

### Production

```
Load Balancer
├── API Server 1 (Docker)
├── API Server 2 (Docker)
└── API Server N (Docker)
    ↓
PostgreSQL Cluster
├── Primary (Write)
└── Replicas (Read)
    ↓
Redis Cache
└── Session & Results Cache
```

## Future Enhancements

1. **Real-time Analysis**: WebSocket support for live analysis
2. **Batch Processing**: Analyze multiple sessions concurrently
3. **Advanced ML Models**: Fine-tuned models for teaching scenarios
4. **Multi-language Support**: Support for non-English sessions
5. **Video Analysis**: Incorporate visual cues from video sessions
6. **Recommendation System**: Suggest improvements to teachers
7. **Trend Analysis**: Aggregate analytics across sessions
8. **A/B Testing**: Compare different analysis approaches
