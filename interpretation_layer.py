import ollama
import json
import time

def load_patient_profile():
    try:
        with open("patient_profile.json") as f:
            return json.load(f)
    except:
        return {"name": "Unknown", "age": "Unknown", "conditions": [],
                "medications": [], "fall_history": "None", "mobility": "Unknown"}

def interpret_fall(danger: dict, features: dict) -> dict:
    profile   = load_patient_profile()
    flags     = danger.get('flags', {})
    active    = [k.replace('_',' ') for k,v in flags.items() if v]
    conf      = danger.get('combined_confidence', 0)
    fall_prob = danger.get('fall_probability', 0)
    spine     = features.get('spine_angle', 0)
    hip_vel   = features.get('hip_velocity', 0)

    prompt = f"""Patient: {profile.get('name')}, age {profile.get('age')}, gender {profile.get('gender','unknown')}.
Conditions: {', '.join(profile.get('conditions', []))}.
Medications: {', '.join(profile.get('medications', []))}.
Fall history: {profile.get('fall_history')}.
Mobility: {profile.get('mobility')}.

Fall detected at {time.strftime('%H:%M:%S')}: {', '.join(active) if active else 'sudden fall'}.
Confidence: {conf:.0%}. Fall probability: {fall_prob:.0%}.
Spine angle: {spine:.1f}°. Hip velocity: {hip_vel:.3f}.

Important: If the person is on the floor and NOT getting up, urgency must be HIGH or CRITICAL.
A person lying flat (body height < 0.15, spine angle < 20) for extended time is always HIGH risk minimum.

You MUST respond with ONLY this JSON, no other text:
{{"risk_assessment": "...", "possible_causes": "...", "immediate_action": "...", "urgency": "LOW or MODERATE or HIGH or CRITICAL", "telegram_message": "max 200 chars plain text alert for family"}}"""

    try:
        response = ollama.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': prompt}]
        )
        raw = response['message']['content'].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        start = raw.find('{')
        end   = raw.rfind('}') + 1
        return json.loads(raw[start:end])
    except Exception as e:
        print(f"[INTERPRETATION ERROR] {e}")
        return {
            "risk_assessment": "Fall detected — assessment unavailable.",
            "possible_causes": "Unable to determine.",
            "immediate_action": "Please check on the patient immediately.",
            "urgency": "HIGH",
            "telegram_message": f"FALL ALERT — Please check on {profile.get('name')} immediately."
        }


if __name__ == "__main__":
    print("=== Interpretation Layer Test ===\n")
    result = interpret_fall(
        {'combined_confidence': 0.82, 'fall_probability': 0.79,
         'flags': {'sudden_collapse': True, 'rapid_descent': True,
                   'prolonged_immobility': False, 'abnormal_posture': False}},
        {'spine_angle': 72.0, 'hip_velocity': 0.9, 'body_height': 0.35}
    )
    print(json.dumps(result, indent=2))
