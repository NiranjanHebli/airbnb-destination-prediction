# 🚀 First Launch Guide

## Option 1 — Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Stop
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Services

| Service  | URL                          |
|----------|------------------------------|
| Frontend | http://localhost:3000        |
| Backend  | http://localhost:4000        |
| ML API   | http://localhost:8000        |
| MongoDB  | mongodb://localhost:27017    |

---

## Option 2 — Run Locally (Without Docker)

### Prerequisites

- Python 3.11+
- Node.js 18+
- MongoDB running locally on port 27017

---

### Step 1 — ML Service

```bash
cd ml-service
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- Health check: http://localhost:8000/health
- API docs: http://localhost:8000/docs

---

### Step 2 — Backend

```bash
cd backend
cp .env.example .env    # edit if needed
npm install
npm run dev             # or: npm start
```

Runs at: http://localhost:4000

---

### Step 3 — Frontend

```bash
cd frontend
cp .env.example .env    # set REACT_APP_BACKEND_URL if needed
npm install
npm start
```

Runs at: http://localhost:3000

> In dev mode, CRA proxy automatically forwards `/api/*` requests to the backend.
