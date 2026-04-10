require("dotenv").config();
const express = require("express");
const cors    = require("cors");
const helmet  = require("helmet");
const morgan  = require("morgan");
const rateLimit = require("express-rate-limit");
const mongoose  = require("mongoose");

const predictRouter = require("./routes/predict");

const app  = express();
const PORT = process.env.PORT || 4000;

// ── Security & Middleware ────────────────────────────────────────────────────
app.use(helmet());
app.use(cors({ origin: process.env.FRONTEND_URL || "*" }));
app.use(express.json());
app.use(morgan("dev"));

// Global rate limiter: 100 requests per 15 min per IP
app.use(
  rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 100,
    standardHeaders: true,
    legacyHeaders: false,
  })
);

// ── Routes ───────────────────────────────────────────────────────────────────
app.get("/health", (_req, res) => res.json({ status: "ok" }));
app.use("/api/predict", predictRouter);

// 404 handler
app.use((_req, res) => res.status(404).json({ error: "Route not found" }));

// Global error handler
app.use((err, _req, res, _next) => {
  console.error(err.stack);
  res.status(500).json({ error: "Internal server error", detail: err.message });
});

// ── Database ─────────────────────────────────────────────────────────────────
const MONGO_URI = process.env.MONGO_URI || "mongodb://localhost:27017/airbnb_predictor";

mongoose
  .connect(MONGO_URI)
  .then(() => {
    console.log("✅  MongoDB connected");
    app.listen(PORT, () =>
      console.log(`✅  Backend running on http://localhost:${PORT}`)
    );
  })
  .catch((err) => {
    console.error("❌  MongoDB connection failed:", err.message);
    process.exit(1);
  });
