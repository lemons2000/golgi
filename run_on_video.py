"""
Run full SynapxeAI pipeline on recorded video files.
Usage: python3 run_on_video.py video1.MP4
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import json
import time
import sys
import os
from dataclasses import dataclass, asdict
from collections import deque
from risk_engine_live import predict_danger
from agent import DecisionAgent

MODEL_PATH = "pose_landmarker.task"

FALL_VELOCITY_THRESHOLD  = 0.08
FALL_SPINE_ANGLE_THRESHOLD = 70
IMMOBILITY_WINDOW_SECONDS  = 10.0
IMMOBILITY_MOVEMENT_THRESHOLD = 0.003

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
    landmarks_in_frame: float

def angle_between(v1, v2):
    v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))))

class VideoFallDetector:
    def __init__(self, fps=30):
        self.fps = fps
        self.feature_history  = deque(maxlen=90)
        self.last_movement_time = 0
        self.last_features    = None
        self.critical_frames  = 0
        self.SUSTAINED_FRAMES = 20

    def extract_features(self, landmarks, frame_shape):
        NOSE=0; LS=11; RS=12; LH=23; RH=24; LA=27; RA=28
        key_lms = [NOSE, LS, RS, LH, RH, LA, RA]
        in_frame = [1 if (landmarks[i].visibility > 0.4 and
                          0.02 < landmarks[i].x < 0.98 and
                          0.02 < landmarks[i].y < 0.98) else 0
                    for i in key_lms]
        landmarks_in_frame = sum(in_frame) / len(key_lms)
        if landmarks_in_frame < 0.7:  # more lenient for video
            return None
        vis = np.mean([landmarks[i].visibility for i in [NOSE,LS,RS,LH,RH]])
        if vis < 0.4: return None
        head_y     = landmarks[NOSE].y
        hip_y      = (landmarks[LH].y + landmarks[RH].y) / 2
        shoulder_y = (landmarks[LS].y  + landmarks[RS].y) / 2
        ankle_y    = (landmarks[LA].y  + landmarks[RA].y) / 2
        body_height = ankle_y - head_y
        hip_center      = np.array([(landmarks[LH].x+landmarks[RH].x)/2, hip_y])
        shoulder_center = np.array([(landmarks[LS].x+landmarks[RS].x)/2, shoulder_y])
        spine_angle = angle_between(shoulder_center - hip_center, np.array([0,-1]))
        hip_vel = head_vel = 0.0
        if self.last_features:
            dt = max(1/self.fps, 1e-3)
            hip_vel  = (hip_y  - self.last_features.hip_y)  / dt
            head_vel = (head_y - self.last_features.head_y) / dt
        return PoseFeatures(time.time(), hip_y, head_y, spine_angle,
                            body_height, hip_vel, head_vel, vis,
                            landmarks_in_frame)

    def detect_events(self, f, video_time):
        flags = {"sudden_collapse": False, "prolonged_immobility": False,
                 "abnormal_posture": False, "rapid_descent": False}
        if f.hip_velocity > FALL_VELOCITY_THRESHOLD and f.spine_angle > FALL_SPINE_ANGLE_THRESHOLD:
            flags["sudden_collapse"] = True
        if f.hip_velocity > FALL_VELOCITY_THRESHOLD * 1.5 and f.spine_angle > 30:
            flags["rapid_descent"] = True
        if len(self.feature_history) > 10:
            avg_spine = np.mean([x.spine_angle for x in list(self.feature_history)[-10:]])
            if avg_spine > 60 and f.body_height < 0.2:
                flags["abnormal_posture"] = True
        if self.last_features:
            movement = abs(f.hip_y - self.last_features.hip_y) + abs(f.head_y - self.last_features.head_y)
            if movement > IMMOBILITY_MOVEMENT_THRESHOLD:
                self.last_movement_time = video_time
        immobility_dur = video_time - self.last_movement_time
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

    def _draw(self, frame, landmarks, danger, features, action):
        H, W = frame.shape[:2]
        CONN = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),
                (23,24),(23,25),(24,26),(25,27),(26,28),(0,11),(0,12)]
        pts = [(int(lm.x*W), int(lm.y*H)) for lm in landmarks]
        for a,b in CONN:
            cv2.line(frame, pts[a], pts[b], (0,255,128), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (255,255,255), -1)
        color = danger['color']
        conf  = danger['combined_confidence']
        level = danger['danger_level']
        bar_w = int(W * 0.25)
        cv2.rectangle(frame, (10,10), (10+bar_w,22), (50,50,50), -1)
        cv2.rectangle(frame, (10,10), (10+int(bar_w*conf),22), color, -1)
        cv2.putText(frame, f"RISK {conf:.0%} [{level}]", (10,42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Agent: {action.upper()}", (10,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)
        cv2.putText(frame, f"Spine: {features.spine_angle:.1f} Hip vel: {features.hip_velocity:.3f}",
                    (10, H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        if level == "CRITICAL":
            cv2.rectangle(frame, (0,H-55),(W,H-20), color, -1)
            reasons = [k.replace('_',' ').upper() for k,v in danger['flags'].items() if v]
            cv2.putText(frame, f"!! {' + '.join(reasons) or 'FALL DETECTED'}",
                        (12,H-30), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255), 2)
        return frame

    def _init_pose(self):
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self.pose = mp_vision.PoseLandmarker.create_from_options(options)

    def process(self, video_path: str):
        self._init_pose()  # reinit for each video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening {video_path}")
            return

        fps      = cap.get(cv2.CAP_PROP_FPS) or 30
        total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = fps
        self.last_movement_time = 0

        agent    = DecisionAgent()
        results_log = []
        frame_num   = 0

        print(f"\nProcessing {os.path.basename(video_path)}")
        print(f"FPS: {fps:.0f} | Total frames: {total} | Duration: {total/fps:.1f}s\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_num += 1
            video_time = frame_num / fps  # seconds into video

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result  = self.pose.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            )

            action = "wait"
            danger = {'danger_level':'LOW','combined_confidence':0,
                      'fall_probability':0,'color':(0,220,80),'flags':{}}

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                features  = self.extract_features(landmarks, frame.shape)

                if features:
                    flags, immob = self.detect_events(features, video_time)
                    cv_risk = self.compute_cv_risk(flags)
                    risk_output = {
                        'features': asdict(features),
                        'flags': flags,
                        'risk_score': cv_risk
                    }
                    danger = predict_danger(risk_output)
                    danger['features'] = asdict(features)

                    if danger['danger_level'] == "CRITICAL":
                        self.critical_frames += 1
                    else:
                        self.critical_frames = 0

                    danger_for_agent = dict(danger)
                    if self.critical_frames < self.SUSTAINED_FRAMES:
                        danger_for_agent['danger_level'] = "LOW"
                        danger_for_agent['combined_confidence'] *= 0.3

                    action = agent.update(danger_for_agent)
                    self.feature_history.append(features)
                    self.last_features = features
                    frame = self._draw(frame, landmarks, danger, features, action)

                else:
                    cv2.putText(frame, "Body not fully in frame", (10,40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,165,255), 2)

            # Progress bar
            H, W = frame.shape[:2]
            progress = frame_num / total if total > 0 else 0
            cv2.rectangle(frame, (0,H-4),(W,H), (30,30,30), -1)
            cv2.rectangle(frame, (0,H-4),(int(W*progress),H), (0,229,204), -1)
            cv2.putText(frame, f"{video_time:.1f}s / {total/fps:.1f}s",
                        (W-150, H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

            # Recording overlay
            import datetime
            ts = datetime.datetime.now().strftime("%d %b %Y  %H:%M:%S")
            cv2.putText(frame, ts, (W-230, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(frame, os.path.basename(video_path), (W-230, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

            mdp_state_val = agent.mdp.state.value if hasattr(agent, "mdp") else "N/A"
            mdp_colors = {
                "NORMAL":      (0, 200, 80),
                "LOW_RISK":    (0, 220, 200),
                "MEDIUM_RISK": (0, 165, 255),
                "HIGH_RISK":   (0, 80, 255),
                "CRITICAL":    (0, 40, 220),
            }
            mdp_col = mdp_colors.get(mdp_state_val, (200,200,200))

            # MDP box — top RIGHT, below timestamp (y=55 to y=100)
            if hasattr(agent, "mdp") and agent.mdp.transition_log:
                last_scores = agent.mdp.transition_log[-1].get("scores", {})
                score_txt = (f"fl={last_scores.get('floor',0):.2f} "
                             f"du={last_scores.get('duration',0):.2f} "
                             f"cf={last_scores.get('conf',0):.2f}")
            else:
                score_txt = "fl=0.00 du=0.00 cf=0.00"
            cv2.rectangle(frame, (W-224, 55),(W-2, 102), (20,20,20), -1)
            cv2.rectangle(frame, (W-224, 55),(W-2, 102), mdp_col, 2)
            cv2.putText(frame, f"MDP: {mdp_state_val}",
                        (W-218, 78), cv2.FONT_HERSHEY_DUPLEX, 0.52, mdp_col, 2)
            cv2.putText(frame, score_txt,
                        (W-218, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180,180,180), 1)

            # Watermark — bottom centre
            cv2.putText(frame, "ARGUS  |  AI Fall Detection System",
                        (W//2 - 155, H-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100,100,100), 1)

            # Telegram flash — top left, below Agent text (y=110 to y=135)
            if action in ["notify", "escalate", "recovery_notify"]:
                agent._last_telegram_time = video_time
            last_tg = getattr(agent, "_last_telegram_time", -99)
            if video_time - last_tg < 3:
                msg_map = {
                    "notify":          "Alert Sent!",
                    "escalate":        "Escalation Sent!",
                    "recovery_notify": "Recovery Sent!",
                }
                flash_msg = msg_map.get(action, "Sent!")
                cv2.rectangle(frame, (8, 108),(185, 135), (20,20,20), -1)
                cv2.rectangle(frame, (8, 108),(185, 135), mdp_col, 2)
                cv2.putText(frame, flash_msg,
                            (14, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.48, mdp_col, 2)

            results_log.append({
                'frame': frame_num,
                'time_s': round(video_time, 2),
                'action': action,
                'level': danger['danger_level'],
                'confidence': round(danger['combined_confidence'], 3),
                'fall_prob': round(danger['fall_probability'], 3)
            })

            cv2.imshow(f"SynapxeAI — {os.path.basename(video_path)}", frame)
            # Press Q to quit, SPACE to pause
            # Stop video5 at 25 seconds
            if os.path.basename(video_path) == "video5.MP4" and video_time >= 25:
                print("[INFO] Stopping video5 at 25s mark")
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord(' '):
                while cv2.waitKey(0) & 0xFF != ord(' '): pass

        cap.release()
        cv2.destroyAllWindows()
        self.pose.close()
        # pose will be reinited on next video

        # Save results
        out_file = video_path.replace('.MP4','.json').replace('.mp4','.json')
        with open(out_file, 'w') as f:
            json.dump(results_log, f, indent=2)

        # Summary
        falls     = sum(1 for r in results_log if r['action'] == 'notify')
        criticals = sum(1 for r in results_log if r['level'] == 'CRITICAL')
        print(f"\n=== Results: {os.path.basename(video_path)} ===")
        print(f"Total frames:    {frame_num}")
        print(f"Critical frames: {criticals} ({criticals/max(frame_num,1):.0%})")
        print(f"Fall alerts:     {falls}")
        print(f"Results saved:   {out_file}")


if __name__ == "__main__":
    videos = sys.argv[1:] if len(sys.argv) > 1 else [
        "video1.MP4", "video2.MP4", "video3.MP4", "video4.MP4", "video5.MP4", "video6.MP4", "video7.MP4", "video8.MP4", "video9.MP4"
    ]
    detector = VideoFallDetector()
    for v in videos:
        path = os.path.join(os.path.expanduser("~/documents/synapxe!"), v)
        if os.path.exists(path):
            detector.process(path)
        else:
            print(f"Not found: {path}")
