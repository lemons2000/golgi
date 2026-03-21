import joblib
import numpy as np
import os

MODEL_PATH  = "risk_model.pkl"
SCALER_PATH = "risk_scaler.pkl"

try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    MODEL_LOADED = True
except:
    MODEL_LOADED = False
    print("[RISK ENGINE] Model not found, using rule-based only")

def cv_to_imu_features(features: dict) -> np.ndarray:
    spine    = features.get('spine_angle', 0)
    hip_vel  = features.get('hip_velocity', 0)
    head_vel = features.get('head_velocity', 0)
    body_h   = features.get('body_height', 2.0)
    vis      = features.get('visibility', 1.0)

    body_h_norm  = np.clip(body_h / 2.5, 0, 1)
    is_horizontal = 1 - body_h_norm
    impact       = abs(hip_vel) + abs(head_vel)
    fall_signal  = impact * (0.5 + 0.5 * is_horizontal)
    spine_norm   = np.clip(spine / 90.0, 0, 1)

    return np.array([
        impact, impact, fall_signal, fall_signal,
        abs(hip_vel), abs(head_vel), spine_norm,
        is_horizontal, fall_signal, impact * spine_norm,
        is_horizontal * spine_norm, fall_signal * spine_norm
    ]).reshape(1, -1)

def predict_danger(risk_output: dict) -> dict:
    features  = risk_output.get('features', {})
    flags     = risk_output.get('flags', {})
    cv_risk   = risk_output.get('risk_score', 0)

    spine    = features.get('spine_angle', 0)
    hip_vel  = features.get('hip_velocity', 0)
    head_vel = features.get('head_velocity', 0)
    body_h   = features.get('body_height', 2.0)
    vis      = features.get('visibility', 1.0)

    # ── Rule-based score ─────────────────────────────────────
    rule_score = 0.0

    # PRIMARY: person is horizontal on floor
    if body_h < 0.15:
        rule_score += 0.75   # strong signal — person is flat
    elif body_h < 0.25:
        rule_score += 0.50

    # SECONDARY: velocity at time of fall
    if abs(hip_vel) > 0.05 or abs(head_vel) > 0.05:
        rule_score += 0.2

    # TERTIARY: spine angle
    if spine > 60:
        rule_score += 0.15
    elif spine > 40:
        rule_score += 0.08

    # CV event flags
    if flags.get('sudden_collapse'):      rule_score += 0.15
    if flags.get('rapid_descent'):        rule_score += 0.10
    if flags.get('abnormal_posture'):     rule_score += 0.10
    if flags.get('prolonged_immobility'): rule_score += 0.20

    rule_score = min(rule_score, 1.0)

    # ── RF model score ───────────────────────────────────────
    rf_prob = 0.0
    if MODEL_LOADED:
        try:
            imu_feats = cv_to_imu_features(features)
            scaled    = scaler.transform(imu_feats)
            rf_prob   = float(model.predict_proba(scaled)[0][1])
        except:
            pass

    # ── Combined score ───────────────────────────────────────
    # Weight rule_score heavily since body_height is reliable
    if MODEL_LOADED:
        combined = (rule_score * 0.65) + (rf_prob * 0.25) + (cv_risk * 0.10)
    else:
        combined = (rule_score * 0.75) + (cv_risk * 0.25)

    combined = min(combined, 1.0)

    # Determine level — lower threshold since body_h is reliable
    if combined >= 0.45:
        level = "CRITICAL"
        color = (0, 40, 220)
    else:
        level = "LOW"
        color = (0, 220, 80)

    return {
        'danger_level':        level,
        'fall_probability':    round(rf_prob, 3),
        'combined_confidence': round(combined, 3),
        'rule_score':          round(rule_score, 3),
        'color':               color,
        'flags':               flags,
    }


if __name__ == "__main__":
    # Simulate person on floor (body_h=0.05, like video5)
    print("=== Standing ===")
    r = predict_danger({'features': {'spine_angle': 5, 'hip_velocity': 0.01,
        'head_velocity': 0.01, 'body_height': 0.55, 'visibility': 0.99},
        'flags': {}, 'risk_score': 0.0})
    print(f"Level: {r['danger_level']} | Combined: {r['combined_confidence']}")

    print("\n=== On floor (like video5 t=10s) ===")
    r = predict_danger({'features': {'spine_angle': 5, 'hip_velocity': 0.01,
        'head_velocity': 0.01, 'body_height': 0.055, 'visibility': 0.95},
        'flags': {}, 'risk_score': 0.0})
    print(f"Level: {r['danger_level']} | Combined: {r['combined_confidence']}")

    print("\n=== Falling (mid-fall) ===")
    r = predict_danger({'features': {'spine_angle': 45, 'hip_velocity': 0.5,
        'head_velocity': 0.4, 'body_height': 0.28, 'visibility': 0.97},
        'flags': {'sudden_collapse': True, 'rapid_descent': True}, 'risk_score': 0.6})
    print(f"Level: {r['danger_level']} | Combined: {r['combined_confidence']}")
