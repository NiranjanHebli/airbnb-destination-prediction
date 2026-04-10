# 🏠 Airbnb Destination Predictor — Full-Stack ML Application

Predict the **top 5 most likely destination countries** for a new Airbnb user, powered by an XGBoost model trained on the Kaggle Airbnb New User Booking dataset.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Browser                         │
│                   React Frontend (:3000)                    │
│      Form → Submit → Display Top-5 Country Predictions      │
└──────────────────────┬──────────────────────────────────────┘
                       │  POST /api/predict
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Node.js Express Backend (:4000)                │
│  • Input validation (express-validator)                     │
│  • Rate limiting / CORS / Helmet security                   │
│  • Calls ML service                                         │
│  • Saves prediction + input to MongoDB                      │
└──────────┬────────────────────────────┬─────────────────────┘
           │  POST /predict             │  mongoose
           ▼                            ▼
┌──────────────────────┐   ┌────────────────────────────────┐
│  FastAPI ML Service  │   │         MongoDB (:27017)        │
│       (:8000)        │   │   Collection: predictions      │
│  • Loads .pkl files  │   │   Stores: input + top5 +       │
│  • Preprocesses input│   │   timestamp + ip + user-agent  │
│  • XGBoost predict   │   └────────────────────────────────┘
│  • Returns top-5     │
│    country codes     │
└──────────────────────┘
```

---

## 📁 Folder Structure

```
airbnb-predictor/
├── ml-service/                  # Python FastAPI ML microservice
│   ├── main.py                  # FastAPI app with /predict + /health
│   ├── production_model.pkl     # Trained XGBoost model
│   ├── preprocessor.pkl         # Fitted ColumnTransformer
│   ├── label_encoder.pkl        # LabelEncoder for country codes
│   ├── requirements.txt
│   └── Dockerfile
│
├── backend/                     # Node.js Express API
│   ├── src/
│   │   ├── index.js             # App entry: Express + MongoDB setup
│   │   ├── routes/
│   │   │   └── predict.js       # POST /api/predict, GET /api/predict/history
│   │   └── models/
│   │       └── Prediction.js    # Mongoose schema
│   ├── .env.example
│   ├── package.json
│   └── Dockerfile
│
├── frontend/                    # React SPA
│   ├── src/
│   │   ├── App.js               # Root component + API call
│   │   ├── App.css              # All styles (Airbnb design system)
│   │   ├── index.js             # React entry point
│   │   └── components/
│   │       ├── Header.js
│   │       ├── PredictionForm.js
│   │       └── PredictionResults.js
│   ├── public/
│   │   └── index.html
│   ├── .env.example
│   ├── package.json
│   └── Dockerfile
│
├── docker-compose.yml           # Orchestrates all 4 services
└── README.md
```

---

## 🧠 Feature Schema

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Float (15–90) | User age |
| `gender` | Categorical | MALE / FEMALE / OTHER / -unknown- |
| `signup_method` | Categorical | basic / facebook / google |
| `device_type` | Categorical | Mac Desktop, iPhone, etc. |
| `total_actions` | Integer ≥ 0 | Total session actions |
| `total_time` | Float ≥ 0 | Total session time in seconds |

**Output countries:** `AU` `CA` `DE` `ES` `FR` `GB` `IT` `NDF` `NL` `PT` `US` `other`

(`NDF` = No Destination Found — user did not book)

---

## 🚀 Running Locally (Without Docker)

### Prerequisites
- Python 3.11+
- Node.js 18+
- MongoDB running locally on port 27017

### 1. ML Service

```bash
cd ml-service
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Verify: http://localhost:8000/health  
Docs:   http://localhost:8000/docs

### 2. Backend

```bash
cd backend
cp .env.example .env        # edit if needed
npm install
npm run dev                 # or: npm start
```

Service runs at http://localhost:4000

### 3. Frontend

```bash
cd frontend
cp .env.example .env        # set REACT_APP_BACKEND_URL if needed
npm install
npm start
```

App runs at http://localhost:3000  
(CRA proxy forwards `/api/*` to backend automatically in dev)

---

## 🐳 Running with Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Stop
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend  | http://localhost:4000 |
| ML API   | http://localhost:8000 |
| MongoDB  | mongodb://localhost:27017 |

---

## 🔧 Environment Variables

### ML Service (env vars or set in docker-compose)
| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `production_model.pkl` | Path to model file |
| `PREPROCESSOR_PATH` | `preprocessor.pkl` | Path to preprocessor |
| `ENCODER_PATH` | `label_encoder.pkl` | Path to label encoder |

### Backend (`backend/.env`)
| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `4000` | Express port |
| `MONGO_URI` | `mongodb://localhost:27017/airbnb_predictor` | MongoDB connection string |
| `ML_SERVICE_URL` | `http://localhost:8000` | FastAPI ML service URL |
| `FRONTEND_URL` | `*` | CORS allowed origin |

### Frontend (`frontend/.env`)
| Variable | Default | Description |
|----------|---------|-------------|
| `REACT_APP_BACKEND_URL` | *(empty — uses proxy)* | Backend URL for production |

---

## 📡 API Reference

### ML Service

#### `GET /health`
```json
{ "status": "ok", "model_loaded": true }
```

#### `POST /predict`
**Request:**
```json
{
  "age": 25,
  "gender": "MALE",
  "signup_method": "basic",
  "device_type": "Mac Desktop",
  "total_actions": 50,
  "total_time": 1200
}
```
**Response:**
```json
{
  "top5": [
    { "country": "US",  "probability": 0.581234 },
    { "country": "NDF", "probability": 0.213456 },
    { "country": "other", "probability": 0.089012 },
    { "country": "FR", "probability": 0.054321 },
    { "country": "IT", "probability": 0.031987 }
  ]
}
```

---

### Node.js Backend

#### `POST /api/predict`
Same request body as ML service. Response adds MongoDB `id` and `saved_at`.

#### `GET /api/predict/history`
Returns the last 20 predictions stored in MongoDB.

---

## ☁️ Deployment

### Render (Recommended — Free Tier)

**ML Service:**
1. New Web Service → connect repo → select `ml-service/` as root
2. Runtime: Python 3 | Build: `pip install -r requirements.txt` | Start: `uvicorn main:app --host 0.0.0.0 --port 8000`
3. Add env vars: `MODEL_PATH`, `PREPROCESSOR_PATH`, `ENCODER_PATH`

**Backend:**
1. New Web Service → select `backend/` as root
2. Runtime: Node | Build: `npm install` | Start: `npm start`
3. Add env vars: `MONGO_URI` (MongoDB Atlas URL), `ML_SERVICE_URL` (Render ML URL)

**Frontend:**
1. New Static Site → select `frontend/` as root
2. Build: `npm install && npm run build` | Publish: `build`
3. Add env: `REACT_APP_BACKEND_URL=https://your-backend.onrender.com`

### Vercel (Frontend only)
```bash
cd frontend
npx vercel --prod
# Set REACT_APP_BACKEND_URL in Vercel dashboard
```

---

## 🐛 Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `502 Bad Gateway` | ML service not running | Start ML service first; check `ML_SERVICE_URL` |
| `MongoServerError: Connection refused` | MongoDB not running | Start MongoDB or check `MONGO_URI` |
| `sklearn version mismatch warning` | Pickle trained on sklearn 1.6.1 | Already patched in `main.py`; use same sklearn version for retraining |
| `422 Unprocessable Entity` | Invalid input values | Check age (15–90), signup_method (basic/facebook/google) |
| `CORS error in browser` | FRONTEND_URL mismatch | Set `FRONTEND_URL` in backend `.env` to your frontend origin |
| Frontend shows "Unable to reach server" | Backend not running or wrong URL | Check `REACT_APP_BACKEND_URL` in frontend `.env` |
| Docker: port already in use | Local service on same port | Stop local service or change port in `docker-compose.yml` |

---

## 🔬 Model Details

| Property | Value |
|----------|-------|
| Algorithm | XGBoost (`multi:softprob`) |
| Trees | 300 |
| Max Depth | 6 |
| Learning Rate | 0.1 |
| Classes | 12 (AU CA DE ES FR GB IT NDF NL PT US other) |
| Metric | NDCG@5 (Kaggle) |
| Preprocessing | ColumnTransformer: StandardScaler (numeric) + OneHotEncoder (categorical) |
