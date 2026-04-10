const express = require("express");
const axios = require("axios");
const { body, validationResult } = require("express-validator");

const Prediction = require("../models/Prediction");

const router = express.Router();
const ML_URL = process.env.ML_SERVICE_URL || "http://localhost:8000";

// ── Validation rules ─────────────────────────────────────────────────────────
const validateInput = [
  body("age")
    .isFloat({ min: 15, max: 90 })
    .withMessage("age must be a number between 15 and 90"),
  body("gender")
    .isString()
    .notEmpty()
    .withMessage("gender is required"),
  body("signup_method")
    .isIn(["basic", "facebook", "google"])
    .withMessage("signup_method must be basic, facebook, or google"),
  body("device_type")
    .isString()
    .notEmpty()
    .withMessage("device_type is required"),
  body("total_actions")
    .isInt({ min: 0 })
    .withMessage("total_actions must be a non-negative integer"),
  body("total_time")
    .isFloat({ min: 0 })
    .withMessage("total_time must be a non-negative number"),
];

// ── POST /api/predict ─────────────────────────────────────────────────────────
router.post("/", validateInput, async (req, res) => {
  // Check express-validator errors
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(422).json({ errors: errors.array() });
  }

  // Coerce to proper numeric types — parseFloat/parseInt can yield NaN from form strings
  const age = parseFloat(req.body.age) || 0;
  const total_actions = parseInt(req.body.total_actions, 10) || 0;
  const total_time = parseFloat(req.body.total_time) || 0;
  const { gender, signup_method, device_type } = req.body;

  // Sanity-check numerics after coercion
  if (!isFinite(age) || age < 15 || age > 90) {
    return res.status(422).json({ error: "age must be a finite number between 15 and 90" });
  }
  if (!isFinite(total_actions) || total_actions < 0) {
    return res.status(422).json({ error: "total_actions must be a non-negative integer" });
  }
  if (!isFinite(total_time) || total_time < 0) {
    return res.status(422).json({ error: "total_time must be a non-negative number" });
  }

  const mlPayload = { age, gender, signup_method, device_type, total_actions, total_time };

  // Call Python ML service
  let mlResponse;
  try {
    mlResponse = await axios.post(
      `${ML_URL}/predict`,
      mlPayload,
      { timeout: 10_000 }
    );
  } catch (err) {
    // Surface the exact FastAPI 422 detail so the frontend shows something useful
    const detail =
      err.response?.data?.detail
        ? JSON.stringify(err.response.data.detail)
        : err.message || "ML service unavailable";
    console.error("ML service error:", err.response?.status, detail);
    return res.status(502).json({ error: "ML service error", detail });
  }

  const { top5 } = mlResponse.data;

  // Persist to MongoDB
  try {
    const record = await Prediction.create({
      input: mlPayload,
      predictions: top5,
      top1_country: top5[0]?.country,
      ip_address: req.ip,
      user_agent: req.headers["user-agent"],
    });

    return res.status(200).json({
      id: record._id,
      top5,
      saved_at: record.createdAt,
    });
  } catch (dbErr) {
    // Still return predictions even if DB write fails
    console.error("DB write error:", dbErr.message);
    return res.status(200).json({ top5, warning: "Result not persisted to DB" });
  }
});

// ── GET /api/predict/history — last 20 predictions ───────────────────────────
router.get("/history", async (_req, res) => {
  try {
    const history = await Prediction.find()
      .sort({ createdAt: -1 })
      .limit(20)
      .select("-ip_address -user_agent");
    return res.json(history);
  } catch (err) {
    return res.status(500).json({ error: "Failed to fetch history" });
  }
});

module.exports = router;
