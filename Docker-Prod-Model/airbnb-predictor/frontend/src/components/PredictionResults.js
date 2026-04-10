import React from "react";

// Country metadata: display name + emoji flag
const COUNTRY_META = {
  US:    { name: "United States",  flag: "🇺🇸" },
  FR:    { name: "France",         flag: "🇫🇷" },
  IT:    { name: "Italy",          flag: "🇮🇹" },
  GB:    { name: "United Kingdom", flag: "🇬🇧" },
  ES:    { name: "Spain",          flag: "🇪🇸" },
  DE:    { name: "Germany",        flag: "🇩🇪" },
  CA:    { name: "Canada",         flag: "🇨🇦" },
  AU:    { name: "Australia",      flag: "🇦🇺" },
  NL:    { name: "Netherlands",    flag: "🇳🇱" },
  PT:    { name: "Portugal",       flag: "🇵🇹" },
  NDF:   { name: "No Destination Found", flag: "❓" },
  other: { name: "Other Country",  flag: "🌍" },
};

const GENDER_LABELS = {
  MALE: "Male", FEMALE: "Female", OTHER: "Other", "-unknown-": "Unspecified",
};

const SIGNUP_LABELS = {
  basic: "Basic (Email)", facebook: "Facebook", google: "Google",
};

export default function PredictionResults({ predictions, input, onReset }) {
  const maxProb = predictions[0]?.probability || 1;

  return (
    <div className="card">
      <div className="results-header">
        <h2 className="card-title">Top 5 Predicted Destinations</h2>
        <p className="card-subtitle">
          Based on the user profile, here are the most likely booking destinations.
        </p>
        <div className="results-meta">
          <span className="tag">👤 Age {input.age}</span>
          <span className="tag">⚧ {GENDER_LABELS[input.gender] || input.gender}</span>
          <span className="tag">🔑 {SIGNUP_LABELS[input.signup_method]}</span>
          <span className="tag">💻 {input.device_type}</span>
          <span className="tag">🖱️ {input.total_actions} actions</span>
          <span className="tag">⏱️ {input.total_time}s</span>
        </div>
      </div>

      <div className="predictions-list">
        {predictions.map((pred, idx) => {
          const meta  = COUNTRY_META[pred.country] || { name: pred.country, flag: "🌐" };
          const pct   = (pred.probability * 100).toFixed(1);
          const barW  = ((pred.probability / maxProb) * 100).toFixed(1);
          const isTop = idx === 0;

          return (
            <div
              key={pred.country}
              className={`prediction-item${isTop ? " top-pick" : ""}`}
            >
              {/* Background probability bar */}
              <div className="prediction-bar" style={{ width: `${barW}%` }} />

              {isTop && <span className="top-pick-badge">Top Pick</span>}

              <span className="prediction-rank">#{idx + 1}</span>

              <span className="prediction-flag">{meta.flag}</span>

              <div className="prediction-info">
                <div className="prediction-country">{meta.name}</div>
                <div className="prediction-code">{pred.country}</div>
              </div>

              <div className="prediction-prob">
                <div className="prob-value">{pct}%</div>
                <div className="prob-bar-track">
                  <div
                    className="prob-bar-fill"
                    style={{ width: `${barW}%` }}
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <button className="btn btn-secondary" onClick={onReset} style={{ width: "100%" }}>
        ← Predict Another User
      </button>
    </div>
  );
}
