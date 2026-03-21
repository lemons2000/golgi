import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import time
import json
import urllib.request
import os
from dataclasses import dataclass, asdict
from collections import deque
from risk_engine_live import predict_danger
from agent import DecisionAgent

MODEL_PATH = "pose_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

FALL_VELOCITY_THRESHOLD = 0.08
FALL_SPINE_ANGLE_THRESHOLD = 70
IMMOBILITY_WINDOW_SECONDS = 10.0
IMMOBILITY_MOVEMENT_THRESHOLD = 0.003
ABNORMAL_POSTURE_SPINE = 60
ABNORMAL_POSTURE_HEIGHT = 0.2

@dataclass
class PoseFeatures:
    timestamp: float
    hip_y: float
    head_y: float
    spine_angle: float
    body_height: float
    hip_velocity: float
    head_velocity: float
    visibility: float

def angle_between(v1, v2):
    v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))))

class FallDetector:
    def __init__(self, fps=30):
        ensure_model()
        self.fps = fps
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose = mp_vision.PoseLandmarker.create_from_options(options)
        self.feature_history = deque(maxlen=90)
        self.last_movement_time = time.time()
        self.last_features = None

    def extract_features(self, landmarks, frame_shape):
        NOSE=0; LS=11; RS=12; LH=23; RH=24; LA=27; RA=28
        vis = np.mean([landmarks[i].visibility for i in [NOSE,LS,RS,LH,RH]])
        if vis < 0.4: return None
        head_y = landmarks[NOSE].y
        hip_y = (landmarks[LH].y + landmarks[RH].y) / 2
        shoulder_y = (landmarks[LS].y + landmarks[RS].y) / 2
        ankle_y = (landmarks[LA].y + landmarks[RA].y) / 2
        body_height = ankle_y - head_y
        hip_center = np.array([(landmarks[LH].x+landmarks[RH].x)/2, hip_y])
        shoulder_center = np.array([(landmarks[LS].x+landmarks[RS].x)/2, shoulder_y])
        spine_angle = angle_between(shoulder_center - hip_center, np.array([0,-1]))
        hip_vel = head_vel = 0.0
        if self.last_features:
            dt = max(1/self.fps, 1e-3)
            hip_vel = (hip_y - self.last_features.hip_y) / dt
            head_vel = (head_y - self.last_features.head_y) / dt
        return PoseFeatures(time.time(), hip_y, head_y, spine_angle,
                            body_height, hip_vel, head_vel, vis)

    def detect_events(self, f):
        flags = {"sudden_collapse": False, "prolonged_immobility": False,
                 "abnormal_posture": False, "rapid_descent": False}
        if f.hip_velocity > FALL_VELOCITY_THRESHOLD and f.spine_angle > FALL_SPINE_ANGLE_THRESHOLD:
            flags["sudden_collapse"] = True
        if f.hip_velocity > FALL_VELOCITY_THRESHOLD * 1.5 and f.spine_angle > 30:
            flags["rapid_descent"] = True
        if len(self.feature_history) > 10:
            avg_spine = np.mean([x.spine_angle for x in list(self.feature_history)[-10:]])
            if avg_spine > ABNORMAL_POSTURE_SPINE and f.body_height < ABNORMAL_POSTURE_HEIGHT:
                flags["abnormal_posture"] = True
        if self.last_features:
            movement = abs(f.hip_y - self.last_features.hip_y) + abs(f.head_y - self.last_features.head_y)
            if movement > IMMOBILITY_MOVEMENT_THRESHOLD:
                self.last_movement_time = f.timestamp
        immobility_dur = f.timestamp - self.last_movement_time
        if immobility_dur > IMMOBILITY_WINDOW_SECONDS:
            flags["prolonged_immobility"] = True
        return flags, immobility_dur

    def compute_cv_risk(self, flags):
        score = 0.0
        if flags["sudden_collapse"]:      score += 0.6
        if flags["rapid_descent"]:        score += 0.3
        if flags["abnormal_posture"]:     score += 0.25
        if flags["prolonged_immobility"]: score += 0.4
        return min(score, 1.0)

    def _draw_landmarks(self, frame, landmarks):
        H, W = frame.shape[:2]
        CONN = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),
                (23,24),(23,25),(24,26),(25,27),(26,28),(0,11),(0,12)]
        pts = [(int(lm.x*W), int(lm.y*H)) for lm in landmarks]
        for a,b in CONN:
            cv2.line(frame, pts[a], pts[b], (0,255,128), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (255,255,255), -1)
        return frame

    def _annotate(self, frame, danger, features, immobility_dur, action):
        H, W = frame.shape[:2]
        color = danger['color']
        conf = danger['combined_confidence']
        level = danger['danger_level']
        bar_w = int(W * 0.25)
        cv2.rectangle(frame, (10,10), (10+bar_w, 22), (50,50,50), -1)
        cv2.rectangle(frame, (10,10), (10+int(bar_w*conf), 22), color, -1)
        cv2.putText(frame, f"RISK {conf:.0%}  [{level}]", (10,42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Fall prob: {danger['fall_probability']:.0%}", (10,65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(frame, f"Agent: {action.upper()}", (10,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        metrics = [f"Spine: {features.spine_angle:.1f}",
                   f"Hip vel: {features.hip_velocity:.3f}",
                   f"Still: {immobility_dur:.1f}s"]
        for i, m in enumerate(metrics):
            cv2.putText(frame, m, (W-210, 22+i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,200,200), 1)
        if level == "CRITICAL":
            cv2.rectangle(frame, (0,H-55), (W,H), color, -1)
            reasons = [k.replace('_',' ').upper() for k,v in danger['flags'].items() if v]
            cv2.putText(frame, f"!! {' + '.join(reasons) if reasons else 'FALL DETECTED'}",
                        (12,H-18), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255), 2)
        return frame

    def process_frame(self, frame, agent):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not results.pose_landmarks:
            return frame, None, "wait"
        landmarks = results.pose_landmarks[0]
        features = self.extract_features(landmarks, frame.shape)
        if not features:
            return frame, None, "wait"
        flags, immobility_dur = self.detect_events(features)
        cv_risk = self.compute_cv_risk(flags)
        risk_output = {'features': asdict(features), 'flags': flags, 'risk_score': cv_risk}
        danger = predict_danger(risk_output)
        danger["features"] = asdict(features)
        action = agent.update(danger)
        self.feature_history.append(features)
        self.last_features = features
        frame = self._draw_landmarks(frame, landmarks)
        frame = self._annotate(frame, danger, features, immobility_dur, action)
        return frame, danger, action

    def release(self):
        self.pose.close()

def run_with_agent():
    detector = FallDetector()
    agent = DecisionAgent()
    critical_buffer = []
    SUSTAINED_FRAMES = 15  # must be critical for ~0.5s before alerting
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Full pipeline running — press Q to quit")
    print("Fall in front of camera to test Telegram alert!")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame, danger, action = detector.process_frame(frame, agent)
        # Stream live risk to dashboard
        if danger:
            with open("live_risk.json", "w") as f:
                import json as _j
                _j.dump({
                    "confidence": danger.get("combined_confidence", 0),
                    "level": danger.get("danger_level", "LOW"),
                    "fall_prob": danger.get("fall_probability", 0),
                    "action": action,
                    "timestamp": __import__("time").time()
                }, f)
        if danger:
            vis = danger.get("features", {}).get("visibility", 1)
            if vis < 0.85:
                danger["danger_level"] = "LOW"
                danger["combined_confidence"] = 0.0
                cv2.putText(frame, "LOW VISIBILITY - SKIPPING", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
        # Only call agent if sustained critical
        if danger:
            if danger["danger_level"] == "CRITICAL":
                critical_buffer.append(1)
            else:
                critical_buffer.clear()
            if len(critical_buffer) < SUSTAINED_FRAMES:
                danger["danger_level"] = "LOW"
                danger["combined_confidence"] *= 0.3
        cv2.imshow("SynapxeAI — Full Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    detector.release()

if __name__ == "__main__":
    run_with_agent()
