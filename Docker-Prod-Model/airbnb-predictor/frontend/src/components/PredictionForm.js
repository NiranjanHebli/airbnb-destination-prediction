import React, { useState } from "react";

const DEVICE_TYPES = [
  "Mac Desktop",
  "Windows Desktop",
  "iPhone",
  "iPad",
  "Android Phone",
  "Android Tablet",
  "Linux Desktop",
  "Chromebook",
  "Desktop (Other)",
  "Android App Unknown Phone/Tablet",
  "Connected TV",
  "SmartTV",
  "Tablet",
  "Windows Phone",
  "-unknown-",
];

const initialForm = {
  age:           "",
  gender:        "MALE",
  signup_method: "basic",
  device_type:   "Mac Desktop",
  total_actions: "",
  total_time:    "",
};

export default function PredictionForm({ onSubmit, loading, error }) {
  const [form, setForm] = useState(initialForm);

  const set = (field) => (e) =>
    setForm((prev) => ({ ...prev, [field]: e.target.value }));

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      age:           parseFloat(form.age),
      gender:        form.gender,
      signup_method: form.signup_method,
      device_type:   form.device_type,
      total_actions: parseInt(form.total_actions, 10),
      total_time:    parseFloat(form.total_time),
    });
  };

  const isValid =
    form.age &&
    parseFloat(form.age) >= 15 &&
    parseFloat(form.age) <= 90 &&
    form.total_actions !== "" &&
    parseInt(form.total_actions, 10) >= 0 &&
    form.total_time !== "" &&
    parseFloat(form.total_time) >= 0;

  return (
    <div className="card">
      <h2 className="card-title">User Profile</h2>
      <p className="card-subtitle">
        Fill in the details below to predict the top 5 destination countries.
      </p>

      <form onSubmit={handleSubmit}>
        <div className="form-grid">

          {/* Age */}
          <div className="form-group">
            <label className="form-label" htmlFor="age">Age</label>
            <input
              id="age"
              className="form-input"
              type="number"
              min="15"
              max="90"
              step="1"
              placeholder="e.g. 28"
              value={form.age}
              onChange={set("age")}
              required
            />
            <span className="form-hint">Between 15 and 90</span>
          </div>

          {/* Gender */}
          <div className="form-group">
            <label className="form-label" htmlFor="gender">Gender</label>
            <div className="select-wrapper">
              <select
                id="gender"
                className="form-select"
                value={form.gender}
                onChange={set("gender")}
              >
                <option value="MALE">Male</option>
                <option value="FEMALE">Female</option>
                <option value="OTHER">Other</option>
                <option value="-unknown-">Prefer not to say</option>
              </select>
            </div>
          </div>

          {/* Signup Method */}
          <div className="form-group">
            <label className="form-label" htmlFor="signup_method">Signup Method</label>
            <div className="select-wrapper">
              <select
                id="signup_method"
                className="form-select"
                value={form.signup_method}
                onChange={set("signup_method")}
              >
                <option value="basic">Basic (Email)</option>
                <option value="facebook">Facebook</option>
                <option value="google">Google</option>
              </select>
            </div>
          </div>

          {/* Device Type */}
          <div className="form-group">
            <label className="form-label" htmlFor="device_type">Device Type</label>
            <div className="select-wrapper">
              <select
                id="device_type"
                className="form-select"
                value={form.device_type}
                onChange={set("device_type")}
              >
                {DEVICE_TYPES.map((d) => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Total Actions */}
          <div className="form-group">
            <label className="form-label" htmlFor="total_actions">Total Actions</label>
            <input
              id="total_actions"
              className="form-input"
              type="number"
              min="0"
              step="1"
              placeholder="e.g. 50"
              value={form.total_actions}
              onChange={set("total_actions")}
              required
            />
            <span className="form-hint">Total session actions performed</span>
          </div>

          {/* Total Time */}
          <div className="form-group">
            <label className="form-label" htmlFor="total_time">Total Time (seconds)</label>
            <input
              id="total_time"
              className="form-input"
              type="number"
              min="0"
              step="1"
              placeholder="e.g. 1200"
              value={form.total_time}
              onChange={set("total_time")}
              required
            />
            <span className="form-hint">Total browsing time in seconds</span>
          </div>

          {/* Submit */}
          <div className="form-group full-width">
            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading || !isValid}
            >
              {loading ? (
                <>
                  <span className="spinner" />
                  Predicting…
                </>
              ) : (
                "✈️  Predict Destinations"
              )}
            </button>
          </div>
        </div>
      </form>

      {error && (
        <div className="error-banner">
          <span>⚠️</span>
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
