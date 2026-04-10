import React, { useState } from "react";
import axios from "axios";
import PredictionForm from "./components/PredictionForm";
import PredictionResults from "./components/PredictionResults";
import Header from "./components/Header";
import "./App.css";

const API_URL =
  process.env.REACT_APP_BACKEND_URL
    ? `${process.env.REACT_APP_BACKEND_URL}/api/predict`
    : "/api/predict";

function App() {
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState(null);
  const [submittedData, setSubmittedData] = useState(null);

  const handleSubmit = async (formData) => {
    setLoading(true);
    setError(null);
    setPredictions(null);

    try {
      const { data } = await axios.post(API_URL, formData, { timeout: 15000 });
      setPredictions(data.top5);
      setSubmittedData(formData);
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.response?.data?.error ||
        (err.response?.data?.errors
          ? err.response.data.errors.map((e) => e.msg).join(", ")
          : null) ||
        "Unable to reach the server. Please try again.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setPredictions(null);
    setError(null);
    setSubmittedData(null);
  };

  return (
    <div className="app">
      <Header />
      <main className="main">
        <div className="container">
          {!predictions ? (
            <PredictionForm onSubmit={handleSubmit} loading={loading} error={error} />
          ) : (
            <PredictionResults
              predictions={predictions}
              input={submittedData}
              onReset={handleReset}
            />
          )}
        </div>
      </main>
      <footer className="footer">
        <p>Airbnb Destination Predictor &mdash; Powered by XGBoost</p>
      </footer>
    </div>
  );
}

export default App;
