import React from "react";

export default function Header() {
  return (
    <header className="header">
      <span className="header-logo">🏠</span>
      <div className="header-text">
        <h1>Airbnb Destination Predictor</h1>
        <p>ML-powered top-5 country recommendations for new users</p>
      </div>
    </header>
  );
}
