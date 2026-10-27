.PHONY: help install test run clean init-db docker-build docker-run

help:
	@echo "QA Agent System - Available Commands"
	@echo "====================================="
	@echo "make install      - Install dependencies"
	@echo "make init-db      - Initialize database"
	@echo "make test         - Run tests"
	@echo "make test-cov     - Run tests with coverage"
	@echo "make run          - Start API server"
	@echo "make analyze      - Analyze example session"
	@echo "make clean        - Clean cache and logs"
	@echo "make docker-build - Build Docker image"
	@echo "make docker-run   - Run Docker container"

install:
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

init-db:
	python main.py init-db

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=. --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

run:
	python main.py server

analyze:
	python main.py analyze example_session.json

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -name "*.egg-info" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf dist/
	rm -rf build/

docker-build:
	docker build -t qa-agent-system:latest .

docker-run:
	docker compose up --build
