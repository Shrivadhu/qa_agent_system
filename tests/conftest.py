"""Test configuration: run against in-memory SQLite so no Postgres is needed."""

import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


@pytest.fixture(scope="session")
def qa_service_session():
    from api.qa_service import QAService
    service = QAService()
    service.db.create_tables()
    return service
