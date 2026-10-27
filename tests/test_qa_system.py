"""Unit tests for QA Agent system."""

import pytest
from models.schemas import SessionAnalysisRequest, ComponentScores
from api.qa_service import QAService


@pytest.fixture
def qa_service(qa_service_session):
    """Fixture for QA service (shared; models load once)."""
    return qa_service_session


@pytest.fixture
def sample_session_data():
    """Sample session data for testing."""
    return {
        "session_id": "test_session_123",
        "transcript": """
            Teacher: Hello! Today we're going to learn about Python variables.
            First, let's understand what a variable is. A variable is like a 
            container that stores data. For example, we can create a variable 
            called 'name' and store a string in it.
            
            Student: I see, so it's like a box that holds information?
            
            Teacher: Exactly! That's a great way to think about it. Let me show 
            you some examples. We can write: name = "John". This creates a 
            variable called name and stores the string "John" in it.
            
            Student: Got it! Can I try creating my own variable?
            
            Teacher: Of course! Go ahead and try creating a variable called 
            'age' that stores your age.
            
            Student: Okay, so I would write: age = 25
            
            Teacher: Perfect! You've got it. Now you understand how to create 
            and use variables in Python. Variables can store different types 
            of data like strings, integers, and floats.
            
            Student: This makes sense now. Thank you for the clear explanation!
        """,
        "promised_skills": ["python_basics"],
        "feedback_data": {
            "free_text_comment": """
                Great session! The teacher explained variables very clearly 
                with good examples. I now understand how to create and use 
                variables in Python. The pace was perfect and the teacher 
                was patient with my questions.
            """
        },
        "participant_ratings": {
            "clarity": 5.0,
            "helpfulness": 5.0,
            "patience": 5.0
        }
    }


def test_keyword_extraction(qa_service, sample_session_data):
    """Test keyword extraction functionality."""
    result = qa_service.keyword_extractor.extract_keywords(
        sample_session_data["transcript"],
        max_keywords=20
    )
    
    assert result.total_keywords > 0
    assert any('variable' in kw['keyword'].lower() for kw in result.keywords)
    assert result.extraction_method in ['tfidf', 'yake', 'combined']


def test_coverage_checking(qa_service, sample_session_data):
    """Test objective coverage checking."""
    # Extract keywords first
    extraction_result = qa_service.keyword_extractor.extract_keywords(
        sample_session_data["transcript"]
    )
    
    # Generate expected keywords
    expected_keywords = qa_service.coverage_checker.generate_expected_keywords(
        sample_session_data["promised_skills"]
    )
    
    # Check coverage
    coverage_report = qa_service.coverage_checker.check_coverage(
        extraction_result.keywords,
        expected_keywords
    )
    
    assert 0.0 <= coverage_report.coverage_percentage <= 100.0
    assert isinstance(coverage_report.covered_topics, list)
    assert isinstance(coverage_report.missing_topics, list)


def test_teaching_quality(qa_service, sample_session_data):
    """Test teaching quality assessment."""
    segment_scores = qa_service.quality_classifier.evaluate_transcript(
        sample_session_data["transcript"]
    )
    
    assert len(segment_scores) > 0
    
    for score in segment_scores:
        assert 0.0 <= score.clarity <= 1.0
        assert 0.0 <= score.engagement <= 1.0
        assert 0.0 <= score.patience <= 1.0
        assert 0.0 <= score.structure <= 1.0
        assert 0.0 <= score.relevance <= 1.0


def test_learning_progression(qa_service, sample_session_data):
    """Test learning progression tracking."""
    result = qa_service.progression_tracker.detect_progression(
        sample_session_data["transcript"],
        "python_basics"
    )
    
    assert 0.0 <= result.progression_score <= 1.0
    assert result.question_count >= 0
    assert result.knowledge_statements_count >= 0
    assert len(result.progression_graph) > 0


def test_sentiment_analysis(qa_service, sample_session_data):
    """Test sentiment analysis."""
    sentiment, score = qa_service.sentiment_analyzer.analyze_sentiment(
        sample_session_data["feedback_data"]["free_text_comment"]
    )
    
    assert sentiment in ['positive', 'neutral', 'negative']
    assert 0.0 <= score <= 1.0


def test_feedback_authenticity(qa_service, sample_session_data):
    """Test feedback authenticity checking."""
    feedback_text = sample_session_data["feedback_data"]["free_text_comment"]
    
    # Template check
    template_score, is_templated = qa_service.authenticity_checker.check_template_match(
        feedback_text
    )
    assert 0.0 <= template_score <= 1.0
    
    # Human-likeness
    human_score, metrics = qa_service.authenticity_checker.calculate_human_likeness(
        feedback_text
    )
    assert 0.0 <= human_score <= 1.0
    assert 'overall' in metrics


def test_feedback_validation(qa_service, sample_session_data):
    """Test cross-validation of feedback against transcript."""
    result = qa_service.cross_validator.validate_feedback(
        sample_session_data["feedback_data"]["free_text_comment"],
        sample_session_data["transcript"]
    )
    
    assert 0.0 <= result['overall_evidence_score'] <= 1.0
    assert result['total_claims'] >= 0
    assert isinstance(result['claim_validations'], list)


def test_verdict_calculation(qa_service):
    """Test verdict calculation."""
    component_scores = ComponentScores(
        scope_completeness=0.85,
        teaching_quality=0.90,
        learning_indicators=0.80,
        feedback_authenticity=0.88,
        participant_satisfaction=0.92,
        compliance=1.0
    )
    
    overall_score = qa_service.verdict_calculator.aggregate_qa_score(
        component_scores
    )
    
    assert 0.0 <= overall_score <= 1.0
    
    verdict = qa_service.verdict_calculator.calculate_verdict(overall_score)
    assert verdict in ['successful', 'partial', 'dispute', 'failed']


def test_full_analysis(qa_service, sample_session_data):
    """Test complete session analysis."""
    request = SessionAnalysisRequest(**sample_session_data)
    
    result = qa_service.analyze_session(request)
    
    assert result.session_id == sample_session_data["session_id"]
    assert result.qa_verdict is not None
    assert 0.0 <= result.overall_quality_score <= 1.0
    assert result.component_scores is not None
    assert result.evidence is not None
    assert result.recommendation is not None


def test_reanalysis_same_session_does_not_crash(qa_service, sample_session_data):
    """Analyzing the same session twice must update, not violate unique keys."""
    request = SessionAnalysisRequest(**sample_session_data)
    first = qa_service.analyze_session(request)
    second = qa_service.analyze_session(request)
    assert first.qa_report_id == second.qa_report_id


def test_extreme_ratings_do_not_break_validation(qa_service, sample_session_data):
    """Ratings outside 0-5 must not crash ComponentScores validation."""
    data = dict(sample_session_data, participant_ratings={"overall": 10.0})
    result = qa_service.analyze_session(SessionAnalysisRequest(**data))
    assert 0.0 <= result.component_scores.participant_satisfaction <= 1.0


def test_tfidf_returns_keywords(qa_service, sample_session_data):
    """TF-IDF used to return [] for every transcript (min_df=2 on one doc)."""
    kws = qa_service.keyword_extractor.nlp_utils.extract_keywords_tfidf(
        sample_session_data["transcript"]
    )
    assert kws


def test_api_endpoints(qa_service, sample_session_data):
    from fastapi.testclient import TestClient
    import api

    api.qa_service = qa_service
    client = TestClient(api.app)

    health_resp = client.get("/health").json()
    assert health_resp.get("status") == "healthy"

    sample = dict(sample_session_data, session_id="api_session_1")
    resp = client.post("/analyze", json=sample)
    assert resp.status_code == 200, resp.text
    assert resp.json()["session_id"] == "api_session_1"

    got = client.get("/report/api_session_1")
    assert got.status_code == 200
    assert client.get("/report/does_not_exist").status_code == 404


def test_compliance_clean(qa_service, sample_session_data):
    """Clean transcript gets full compliance score."""
    request = SessionAnalysisRequest(**sample_session_data)
    result = qa_service.analyze_session(request)
    assert result.component_scores.compliance == 1.0


def test_compliance_violations(qa_service, sample_session_data):
    """Transcripts with contact sharing, off-platform payments, or abusive language are penalized."""
    dirty_transcript = (
        "Teacher: Hello student, please contact me directly on WhatsApp +1-555-987-6543 or email teacher@external.com. "
        "Do not pay through this platform, send money to my paypal or bitcoin wallet directly. You are an idiot if you pay full price! "
        "Student: Okay I will message you on telegram."
    )
    data = dict(sample_session_data, session_id="dirty_session_01", transcript=dirty_transcript)
    result = qa_service.analyze_session(SessionAnalysisRequest(**data))
    assert result.component_scores.compliance < 0.5
    assert "Compliance violations:" in result.evidence
    assert any(term in result.evidence.lower() for term in ["whatsapp", "email", "payment", "abusive"])


def test_plagiarism_against_past_comments(qa_service, sample_session_data):
    """When a comment is stored, subsequent session with identical feedback is flagged for plagiarism."""
    unique_feedback_text = (
        "Unique commentary 987: The instructor explained concepts with extraordinary precision "
        "and guided through intricate debugging steps very patiently."
    )
    first_data = dict(
        sample_session_data,
        session_id="plag_session_01",
        feedback_data={"free_text_comment": unique_feedback_text}
    )
    first_result = qa_service.analyze_session(SessionAnalysisRequest(**first_data))
    
    # Second session with the exact same feedback comment
    second_data = dict(
        sample_session_data,
        session_id="plag_session_02",
        transcript=sample_session_data["transcript"] + " Let us review more code.",
        feedback_data={"free_text_comment": unique_feedback_text}
    )
    second_result = qa_service.analyze_session(SessionAnalysisRequest(**second_data))
    
    # Plagiarism check should detect similarity > 0.90 against the past session in DB
    assert second_result.component_scores.feedback_authenticity < first_result.component_scores.feedback_authenticity
    assert "possible copied feedback" in second_result.evidence.lower()


def test_custom_skill_keywords_and_taxonomy(qa_service, sample_session_data):
    """Taxonomy supports loaded skills and custom request-level skill_keywords."""
    checker = qa_service.coverage_checker
    assert len(checker.skill_taxonomy) >= 9
    assert "docker_containers" in checker.skill_taxonomy
    assert "sql_databases" in checker.skill_taxonomy

    # Test custom keywords passed in request
    custom_keywords = {"quantum_computing": ["qubit", "superposition", "entanglement"]}
    expected = checker.generate_expected_keywords(["quantum_computing"], custom_skill_keywords=custom_keywords)
    assert "qubit" in expected
    assert "superposition" in expected


def test_api_key_auth_and_skills_endpoint(qa_service, sample_session_data, monkeypatch):
    """API key enforcement: 401 without key, 200 with key, /health open, /skills works."""
    from fastapi.testclient import TestClient
    import api

    api.qa_service = qa_service
    client = TestClient(api.app)

    # When API_KEY is set in environment
    monkeypatch.setenv("API_KEY", "secret-test-key-12345")
    
    # /health is always open
    assert client.get("/health").status_code == 200

    # /skills requires API key
    assert client.get("/skills").status_code == 401
    skills_resp = client.get("/skills", headers={"X-API-Key": "secret-test-key-12345"})
    assert skills_resp.status_code == 200
    assert "python_basics" in skills_resp.json()["skills"]

    # /analyze requires API key
    sample = dict(sample_session_data, session_id="auth_test_session")
    assert client.post("/analyze", json=sample).status_code == 401
    
    auth_resp = client.post("/analyze", json=sample, headers={"X-API-Key": "secret-test-key-12345"})
    assert auth_resp.status_code == 200
    assert auth_resp.json()["session_id"] == "auth_test_session"


def test_batch_analysis(qa_service, sample_session_data):
    """Batch analysis endpoint processes multiple sessions in one request."""
    from fastapi.testclient import TestClient
    import api

    api.qa_service = qa_service
    client = TestClient(api.app)

    batch_payload = {
        "sessions": [
            dict(sample_session_data, session_id="batch_sess_1"),
            dict(sample_session_data, session_id="batch_sess_2")
        ]
    }
    resp = client.post("/analyze/batch", json=batch_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert data["successful"] == 2
    assert data["failed"] == 0
    assert len(data["results"]) == 2


def test_list_reports_and_summary(qa_service, sample_session_data):
    """Verify listing QA reports and querying statistical summary."""
    from fastapi.testclient import TestClient
    import api

    api.qa_service = qa_service
    client = TestClient(api.app)

    # List reports
    list_resp = client.get("/reports?limit=10")
    assert list_resp.status_code == 200
    reports = list_resp.json()
    assert isinstance(reports, list)
    assert len(reports) > 0

    # Summary
    summary_resp = client.get("/reports/summary")
    assert summary_resp.status_code == 200
    summary = summary_resp.json()
    assert summary["total_sessions_analyzed"] >= len(reports)
    assert "verdict_distribution" in summary
    assert "average_scores" in summary
    assert "overall_quality" in summary["average_scores"]


def test_transcript_direct_match(qa_service):
    """Direct verbatim transcript match ensures spoken topics are recognized."""
    checker = qa_service.coverage_checker
    extracted_keywords = [{"keyword": "unrelated", "score": 0.9}]
    expected_keywords = ["indentation", "syntax"]
    
    # Without transcript: missing
    rep_without = checker.check_coverage(extracted_keywords, expected_keywords)
    assert "indentation" in rep_without.missing_topics

    # With transcript: covered
    spoken_transcript = "Teacher: Remember that python indentation and correct syntax are essential."
    rep_with = checker.check_coverage(extracted_keywords, expected_keywords, transcript=spoken_transcript)
    assert "indentation" in rep_with.covered_topics
    assert "syntax" in rep_with.covered_topics
    assert rep_with.coverage_percentage == 100.0


def test_nlp_asr_question_and_understanding_detection(qa_service):
    """NLP utils detect speech-to-text questions lacking question marks and rich understanding cues."""
    nlp = qa_service.keyword_extractor.nlp_utils

    asr_transcript = (
        "can you explain why this function returns None. "
        "what is the difference between list and tuple. "
        "how do we write a loop here. "
        "now i understand the logic completely. "
        "that makes total sense. "
        "i see what you mean."
    )
    
    questions = nlp.detect_questions(asr_transcript)
    assert len(questions) >= 3

    understanding = nlp.detect_statements_of_understanding(asr_transcript)
    assert len(understanding) >= 3


def test_sentiment_negation_handling(qa_service):
    """Sentiment analyzer correctly handles negations (e.g. 'not helpful' is negative)."""
    sentiment_analyzer = qa_service.sentiment_analyzer
    
    neg_comment = "The tutor was not helpful, and the explanations were not clear."
    sentiment, score = sentiment_analyzer.analyze_sentiment(neg_comment)
    assert sentiment == "negative"
    assert score <= 0.5

    pos_comment = "The tutor was extremely thorough and the session was not bad at all."
    sentiment_pos, score_pos = sentiment_analyzer.analyze_sentiment(pos_comment)
    assert sentiment_pos == "positive"
    assert score_pos >= 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

