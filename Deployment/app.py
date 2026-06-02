# ============================================================
#  app.py — NeuralGuard Flask Backend
#  Run:  python app.py
#  API:  POST http://0.0.0.0:5001/predict
#        GET  http://0.0.0.0:5001/health
#        POST http://0.0.0.0:5001/reset
# ============================================================

import os
import sys
import numpy as np
import tensorflow as tf
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import deque

app = Flask(__name__)
CORS(app)  # FIX: Enable CORS so dashboard can call API from any origin

# ── Config ────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
ANOMALY_THRESHOLD = 0.025   # τ = mean + 2σ on training normal MSE
SEQUENCE_LENGTH   = 10      # LSTM timestep window
N_BASE_FEATURES   = 5       # raw sensor count
N_TOTAL_FEATURES  = 8       # 5 raw + 3 engineered

# FIX: Sensor valid ranges for input validation
SENSOR_RANGES = {
    "air_t":   (250.0, 350.0),
    "proc_t":  (250.0, 370.0),
    "rpm":     (500.0,  5000.0),
    "torque":  (0.0,   200.0),
    "wear":    (0.0,   500.0),
}

# Rolling buffer for LSTM sequence
data_buffer = deque(maxlen=SEQUENCE_LENGTH)

print("\n" + "=" * 55)
print("⚡  NeuralGuard — Starting Backend")
print("=" * 55)

# ── Load Artifacts ────────────────────────────────────────────
try:
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))
    # ── Critical contract check ──────────────────────────────────────────────
    # scaler.joblib was fit on the 5 RAW sensor features only (air_t, proc_t,
    # rpm, torque, wear).  The 3 engineered features are appended AFTER scaling,
    # unscaled, exactly as the model was trained.  If someone replaces this file
    # with a different scaler the assertion below will catch the mismatch immediately.
    assert scaler.n_features_in_ == N_BASE_FEATURES, (
        f"scaler.joblib expects {scaler.n_features_in_} features but app.py "
        f"passes {N_BASE_FEATURES}. Re-fit the scaler on the 5 raw features."
    )
    print("✅  StandardScaler loaded")
except AssertionError as e:
    print(f"❌  Scaler contract violated: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌  scaler.joblib not found: {e}")
    sys.exit(1)

try:
    autoencoder = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "autoencoder_model.h5"), compile=False
    )
    print("✅  Autoencoder loaded")
except Exception as e:
    print(f"❌  autoencoder_model.h5 not found: {e}")
    sys.exit(1)

try:
    lstm_model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "best_lstm_attention_model_v2.keras"), compile=False
    )
    print("✅  LSTM+Attention loaded")
except Exception as e:
    print(f"❌  best_lstm_attention_model_v2.keras not found: {e}")
    sys.exit(1)

print("=" * 55 + "\n")


# ── Helpers ───────────────────────────────────────────────────
def get_robust_value(data: dict, primary_key: str, secondary_key: str) -> float:
    """Accept both full column names and short aliases."""
    val = data.get(primary_key)
    if val is None:
        val = data.get(secondary_key)
    if val is None:
        return 0.0
    return float(val)


def validate_inputs(a_t, p_t, rpm, trq, t_w) -> list[str]:
    """
    FIX: Validate that sensor values are within physically realistic bounds.
    Returns a list of warning strings (empty = all OK).
    """
    warnings = []
    vals = {"air_t": a_t, "proc_t": p_t, "rpm": rpm, "torque": trq, "wear": t_w}
    for key, val in vals.items():
        lo, hi = SENSOR_RANGES[key]
        if not (lo <= val <= hi):
            warnings.append(f"{key}={val} outside expected range [{lo}, {hi}]")
    # FIX: Process temperature must be >= air temperature (thermodynamics)
    if p_t < a_t:
        warnings.append(f"proc_t ({p_t}) < air_t ({a_t}): thermodynamically implausible")
    return warnings


def engineer_features(a_t: float, p_t: float, rpm: float, trq: float, t_w: float):
    """
    FIX: Centralised feature engineering matching Notebook 02 exactly.
    Returns (temp_diff, stress_index, torque_wear).
    """
    temp_diff    = p_t - a_t                # thermal differential
    stress_index = trq / (rpm + 1e-5)       # mechanical load proxy
    torque_wear  = trq * t_w                # compound degradation term
    return temp_diff, stress_index, torque_wear


# ── /predict endpoint ─────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        json_data = request.get_json(force=True, silent=True)
        if not json_data:
            return jsonify({"error": "No JSON body received"}), 400

        # STEP 1 — Extract 5 base raw features (both key formats accepted)
        a_t = get_robust_value(json_data, "Air temperature [K]",     "air_t")
        p_t = get_robust_value(json_data, "Process temperature [K]", "proc_t")
        rpm = get_robust_value(json_data, "Rotational speed [rpm]",  "rpm")
        trq = get_robust_value(json_data, "Torque [Nm]",             "torque")
        t_w = get_robust_value(json_data, "Tool wear [min]",         "wear")

        # STEP 1b — FIX: Input validation (non-blocking; surface warnings in response)
        input_warnings = validate_inputs(a_t, p_t, rpm, trq, t_w)

        # STEP 2 — Scale the 5 base features with the pre-fitted scaler
        # FIX: scaler was trained on 5 raw features only — keep this consistent
        base_features = np.array([[a_t, p_t, rpm, trq, t_w]])
        scaled_base   = scaler.transform(base_features)  # shape (1, 5)

        # STEP 3 — Engineer 3 additional features (unscaled, as in notebook 02)
        f6, f7, f8 = engineer_features(a_t, p_t, rpm, trq, t_w)

        # STEP 4 — Concatenate → 8-feature vector  (shape: (8,))
        final_features = np.hstack([scaled_base[0], [f6, f7, f8]])

        # STEP 5 — Append to rolling sequence buffer (deque maxlen=10)
        data_buffer.append(final_features)

        # STEP 6 — Autoencoder anomaly score (MSE on 8-feature vector)
        feat_2d        = final_features.reshape(1, N_TOTAL_FEATURES)
        reconstruction = autoencoder.predict(feat_2d, verbose=0)
        mse = float(np.mean(np.power(feat_2d - reconstruction, 2)))

        # STEP 7 — LSTM prediction (only when buffer has exactly SEQUENCE_LENGTH entries)
        fail_prob = 0.0
        if len(data_buffer) == SEQUENCE_LENGTH:
            seq       = np.array(data_buffer).reshape(1, SEQUENCE_LENGTH, N_TOTAL_FEATURES)
            fail_prob = float(lstm_model.predict(seq, verbose=0)[0][0])
            # FIX: Clamp output to [0, 1] — sigmoid can return values very slightly outside
            fail_prob = float(np.clip(fail_prob, 0.0, 1.0))

        status = "Active" if len(data_buffer) == SEQUENCE_LENGTH else "Stabilizing"

        print(
            f"📥  MSE={mse:.5f}  AE={'⚠️' if mse > ANOMALY_THRESHOLD else '✅'}  "
            f"P(fail)={fail_prob:.2%}  [{status}]  buf={len(data_buffer)}/10",
            flush=True,
        )

        response = {
            "failure_probability": round(fail_prob, 4),
            "anomaly_flag":        bool(mse > ANOMALY_THRESHOLD),
            "anomaly_mse":         round(mse, 6),
            "status":              status,
            "buffer_size":         len(data_buffer),
        }
        # FIX: Surface input warnings to caller (useful for debugging bad data)
        if input_warnings:
            response["input_warnings"] = input_warnings

        return jsonify(response), 200

    except Exception as e:
        print(f"❌  Pipeline error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


# ── /health endpoint ──────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """FIX: Added model_loaded flags so dashboard can confirm backend state."""
    return jsonify({
        "status":         "ok",
        "threshold":      ANOMALY_THRESHOLD,
        "sequence_length": SEQUENCE_LENGTH,
        "buffer_size":    len(data_buffer),
        "models_loaded": {
            "scaler":      True,
            "autoencoder": True,
            "lstm":        True,
        },
    }), 200


# ── /reset endpoint ───────────────────────────────────────────
@app.route("/reset", methods=["POST"])
def reset():
    """
    FIX: New endpoint to clear the rolling buffer.
    Useful when switching machines or starting a new monitoring session
    so stale timesteps from the previous session don't contaminate predictions.
    """
    data_buffer.clear()
    return jsonify({"status": "buffer cleared", "buffer_size": 0}), 200


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🌐  Flask API → http://0.0.0.0:5001")
    print("    POST /predict  |  GET /health  |  POST /reset\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
