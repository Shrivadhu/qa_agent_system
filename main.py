"""Main entry point for QA Agent system."""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from api import start_server
from database import DatabaseRepository
from utils.logger import log


def init_database():
    """Initialize database tables."""
    log.info("Initializing database...")
    db = DatabaseRepository()
    db.create_tables()
    log.info("Database initialized successfully")


def run_server():
    """Run the API server."""
    start_server()


def run_analysis(session_file: str):
    """Run analysis on a session from file."""
    import json
    from models.schemas import SessionAnalysisRequest
    from api.qa_service import QAService
    
    log.info(f"Loading session data from: {session_file}")
    
    with open(session_file, 'r') as f:
        session_data = json.load(f)
    
    request = SessionAnalysisRequest(**session_data)
    
    log.info("Initializing QA service...")
    qa_service = QAService()
    
    log.info("Running analysis...")
    result = qa_service.analyze_session(request)
    
    log.info("=" * 80)
    log.info("QA ANALYSIS RESULTS")
    log.info("=" * 80)
    log.info(f"Session ID: {result.session_id}")
    log.info(f"Verdict: {result.qa_verdict.value.upper()}")
    log.info(f"Overall Score: {result.overall_quality_score:.2f}")
    log.info(f"Recommendation: {result.recommendation.value.upper()}")
    log.info("")
    log.info("Component Scores:")
    log.info(f"  - Scope Completeness: {result.component_scores.scope_completeness:.2f}")
    log.info(f"  - Teaching Quality: {result.component_scores.teaching_quality:.2f}")
    log.info(f"  - Learning Indicators: {result.component_scores.learning_indicators:.2f}")
    log.info(f"  - Feedback Authenticity: {result.component_scores.feedback_authenticity:.2f}")
    log.info(f"  - Participant Satisfaction: {result.component_scores.participant_satisfaction:.2f}")
    log.info(f"  - Compliance: {result.component_scores.compliance:.2f}")
    log.info("")
    log.info(f"Evidence: {result.evidence}")
    log.info("=" * 80)
    
    return result


def run_batch_analysis(path_arg: str):
    """Run batch analysis on a JSON list file or a directory of session files."""
    import json
    from pathlib import Path
    from models.schemas import SessionAnalysisRequest
    from api.qa_service import QAService

    p = Path(path_arg)
    session_files = []
    sessions_data = []

    if p.is_dir():
        session_files = list(p.glob("*.json"))
        for sf in session_files:
            try:
                with open(sf, 'r', encoding='utf-8') as f:
                    sessions_data.append(json.load(f))
            except Exception as e:
                log.error(f"Could not load {sf}: {e}")
    elif p.is_file():
        with open(p, 'r', encoding='utf-8') as f:
            content = json.load(f)
            if isinstance(content, list):
                sessions_data = content
            else:
                sessions_data = [content]

    if not sessions_data:
        log.warning(f"No session JSON data found at: {path_arg}")
        return

    log.info(f"Loaded {len(sessions_data)} sessions for batch analysis. Initializing QA Service...")
    qa_service = QAService()

    success_count = 0
    for idx, s_data in enumerate(sessions_data, 1):
        try:
            req = SessionAnalysisRequest(**s_data)
            res = qa_service.analyze_session(req)
            log.info(f"[{idx}/{len(sessions_data)}] {req.session_id} -> {res.qa_verdict.value.upper()} (score: {res.overall_quality_score:.2f})")
            success_count += 1
        except Exception as e:
            log.error(f"[{idx}/{len(sessions_data)}] Failed to analyze session: {e}")

    log.info(f"Batch completed: {success_count}/{len(sessions_data)} sessions analyzed successfully.")


def show_summary():
    """Display aggregate statistics of all analyzed sessions."""
    from database import DatabaseRepository
    db = DatabaseRepository()
    summary = db.get_qa_summary()
    
    log.info("=" * 60)
    log.info("QA AGENT SYSTEM - AGGREGATE SUMMARY")
    log.info("=" * 60)
    log.info(f"Total Sessions Analyzed: {summary['total_sessions_analyzed']}")
    log.info(f"Verdict Breakdown: {summary['verdict_distribution']}")
    log.info(f"Recommendation Breakdown: {summary['recommendation_distribution']}")
    log.info("Average Component Scores:")
    for comp, score in summary['average_scores'].items():
        log.info(f"  - {comp}: {score:.3f}")
    log.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="QA Agent Post-Session Analysis System"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Init database command
    subparsers.add_parser(
        'init-db',
        help='Initialize database tables'
    )
    
    # Run server command
    subparsers.add_parser(
        'server',
        help='Start the API server'
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze a session from JSON file'
    )
    analyze_parser.add_argument(
        'session_file',
        help='Path to session JSON file'
    )

    # Analyze-batch command
    batch_parser = subparsers.add_parser(
        'analyze-batch',
        help='Analyze multiple sessions from a JSON array file or directory'
    )
    batch_parser.add_argument(
        'path',
        help='Path to JSON file or folder containing session files'
    )

    # Summary command
    subparsers.add_parser(
        'summary',
        help='Show aggregate summary metrics of stored QA reports'
    )
    
    args = parser.parse_args()
    
    if args.command == 'init-db':
        init_database()
    elif args.command == 'server':
        run_server()
    elif args.command == 'analyze':
        run_analysis(args.session_file)
    elif args.command == 'analyze-batch':
        run_batch_analysis(args.path)
    elif args.command == 'summary':
        show_summary()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
