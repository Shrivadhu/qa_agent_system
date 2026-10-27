# Deployment Guide: QA Agent Post-Session Analysis System

This document provides step-by-step instructions for deploying the QA Agent Post-Session Analysis system to production.

---

## Architecture Overview

- **Web / API Server**: FastAPI running via Uvicorn.
- **Database**: PostgreSQL (Production) or SQLite (Development / Lightweight).
- **Cache (Optional)**: Redis for embedding and keyword extraction caching.
- **ML / NLP Engine**: Sentence-Transformers, HuggingFace Transformers, NLTK, YAKE, Scikit-Learn.
- **Hardware Requirement**: Minimum 2 GB RAM (4 GB recommended if running neural transformer models simultaneously).

---

## Option 1: Docker Compose (Recommended for VPS / Dedicated Server)

Docker Compose bundles PostgreSQL 14, Redis 7, and the QA Agent API container with health checks and non-root security.

### 1. Configure Environment
Copy the example environment file and configure secure secrets:
```bash
cp .env.example .env
```
Edit `.env` to set strong values for `DB_PASSWORD` and `API_KEY`:
```ini
DB_PASSWORD=replace_with_strong_random_password_xyz123
API_KEY=your_production_secret_key_98765
CORS_ORIGINS=*
```

### 2. Build and Launch
```bash
docker compose up -d --build
```

### 3. Verify Deployment
Check service status and container health:
```bash
docker compose ps
curl http://localhost:8000/health
```
Expected response:
```json
{"status": "healthy", "database": "connected"}
```

### 4. Setup HTTPS Reverse Proxy (Caddy or Nginx)

#### With Caddy (Automatic Let's Encrypt SSL):
Create `/etc/caddy/Caddyfile`:
```caddy
qa-api.yourdomain.com {
    reverse_proxy localhost:8000
}
```
Reload Caddy:
```bash
sudo systemctl reload caddy
```

#### With Nginx:
```nginx
server {
    listen 80;
    server_name qa-api.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```
Enable SSL using Certbot:
```bash
sudo certbot --nginx -d qa-api.yourdomain.com
```

---

## Option 2: Render Blueprint (One-Click Cloud Deployment)

A `render.yaml` blueprint is included in the project repository.

1. Push this repository to GitHub or GitLab.
2. In the Render Dashboard, navigate to **Blueprints** -> **New Blueprint Instance**.
3. Select your repository.
4. Render will automatically detect `render.yaml` and provision:
   - Managed **PostgreSQL** database instance (`qa-agent-postgres`).
   - Web service (`qa-agent-system`) with automatic `DATABASE_URL` linking, health checks, and generated `API_KEY`.
5. Click **Apply**. Once built, the service will be available on your `onrender.com` subdomain.

---

## Option 3: Direct VPS / Linux Server (Systemd + Python Virtualenv)

### 1. System Requirements & Packages
```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip postgresql postgresql-contrib
```

### 2. PostgreSQL Setup
```bash
sudo -u postgres psql
```
```sql
CREATE DATABASE qa_agent_db;
CREATE USER qa_agent_user WITH ENCRYPTED PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE qa_agent_db TO qa_agent_user;
\q
```

### 3. Project Setup
```bash
cd /opt
sudo git clone <your-repo-url> qa_agent_system
cd qa_agent_system
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Download required NLTK tokenizers
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### 4. Create Systemd Service
Create `/etc/systemd/system/qa-agent.service`:
```ini
[Unit]
Description=QA Agent Post-Session Analysis API
After=network.target postgresql.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/qa_agent_system
Environment="PATH=/opt/qa_agent_system/venv/bin"
EnvironmentFile=/opt/qa_agent_system/.env
ExecStart=/opt/qa_agent_system/venv/bin/python main.py server
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable qa-agent
sudo systemctl start qa-agent
sudo systemctl status qa-agent
```

---

## Option 4: Local / Offline Development (SQLite)

For local development or testing without Postgres:
```bash
# Set SQLite database URL
export DATABASE_URL="sqlite:///./dev.db"

# Run tests
python -m pytest tests/ -v

# Run analysis on example session
python main.py analyze example_session.json

# Run local API server
python main.py server
```
Tables are automatically created on startup.

---

## API Testing & Verification

### 1. Health Check (Open, no auth required)
```bash
curl http://localhost:8000/health
```

### 2. List Supported Skills Taxonomy
```bash
curl -H "X-API-Key: your_secret_api_key_here" http://localhost:8000/skills
```

### 3. Analyze Session
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret_api_key_here" \
  -d @example_session.json
```

### 4. Retrieve Saved QA Report
```bash
curl -H "X-API-Key: your_secret_api_key_here" http://localhost:8000/report/session_example_001
```

---

## Troubleshooting

- **DetachedInstanceError**: Resolved in repository session management with `expire_on_commit=False`.
- **Unique Constraint Violation**: Re-analyzing the same `session_id` updates the existing record automatically.
- **NLTK punkt_tab missing**: Handled automatically in `NLPUtils._ensure_nltk_data()` and Dockerfile.
- **Missing HuggingFace Models in air-gapped environments**: The system automatically switches to TF-IDF cosine similarity and sentiment lexicons gracefully without crashing.
