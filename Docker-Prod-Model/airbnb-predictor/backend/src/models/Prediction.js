const mongoose = require("mongoose");

const predictionSchema = new mongoose.Schema(
  {
    input: {
      age:           { type: Number, required: true },
      gender:        { type: String, required: true },
      signup_method: { type: String, required: true },
      device_type:   { type: String, required: true },
      total_actions: { type: Number, required: true },
      total_time:    { type: Number, required: true },
    },
    predictions: [
      {
        country:     { type: String, required: true },
        probability: { type: Number, required: true },
      },
    ],
    top1_country: { type: String },
    ip_address:   { type: String },
    user_agent:   { type: String },
  },
  { timestamps: true }
);

// Index for fast lookups by top prediction and time
predictionSchema.index({ top1_country: 1 });
predictionSchema.index({ createdAt: -1 });

module.exports = mongoose.model("Prediction", predictionSchema);
