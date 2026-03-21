"""
Runs all videos through ARGUS and saves annotated output as MP4 files.
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import json
import time
import os
import datetime
from dataclasses import dataclass, asdict
from collections import deque
from risk_engine_live import predict_danger
from agent import DecisionAgent

MODEL_PATH = "pose_landmarker.task"
FALL_VELOCITY_THRESHOLD       = 0.08
FALL_SPINE_ANGLE_THRESHOLD    = 70
IMMOBILITY_WINDOW_SECONDS     = 10.0
IMMOBILITY_MOVEMENT_THRESHOLD = 0.003

VIDEOS = [
    "video1.MP4", "video2.MP4", "video3.MP4", "video4.MP4",
    "video5.MP4", "video6.MP4", "video7.MP4", "video8.MP4", "video9.MP4"
]

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

class VideoProcessor:
    def __init__(self, fps=30):
        self.fps = fps
        self._init_pose()
        self.feature_history  = deque(maxlen=90)
        self.last_movement_time = 0
        self.last_features    = None
        self.critical_frames  = 0
        self.low_frames       = 0
        self.SUSTAINED_FRAMES = 1

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

    def extract_features(self, landmarks, frame_shape):
        NOSE=0; LS=11; RS=12; LH=23; RH=24; LA=27; RA=28
        key_lms = [NOSE, LS, RS, LH, RH, LA, RA]
        in_frame = [1 if (landmarks[i].visibility > 0.4 and
                          0.02 < landmarks[i].x < 0.98 and
                          0.02 < landmarks[i].y < 0.98) else 0
                    for i in key_lms]
        landmarks_in_frame = sum(in_frame) / len(key_lms)
        if landmarks_in_frame < 0.7: return None
        vis = np.mean([landmarks[i].visibility for i in [NOSE,LS,RS,LH,RH]])
        if vis < 0.4: return None
        head_y      = landmarks[NOSE].y
        hip_y       = (landmarks[LH].y + landmarks[RH].y) / 2
        shoulder_y  = (landmarks[LS].y + landmarks[RS].y) / 2
        ankle_y     = (landmarks[LA].y + landmarks[RA].y) / 2
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
                            body_height, hip_vel, head_vel, vis, landmarks_in_frame)

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

    def draw(self, frame, landmarks, danger, features, action, video_time,
             video_path, agent, total_frames, frame_num):
        H, W = frame.shape[:2]

        # Scale factor — normalise to 848p baseline
        s = H / 848.0
        def fs(x): return max(0.3, x * s)   # font scale
        def px(x): return int(x * s)         # pixel position
        def th(x): return max(1, int(x * s)) # thickness

        # Skeleton
        CONN = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),
                (23,24),(23,25),(24,26),(25,27),(26,28),(0,11),(0,12)]
        pts = [(int(lm.x*W), int(lm.y*H)) for lm in landmarks]
        for a,b in CONN:
            cv2.line(frame, pts[a], pts[b], (0,255,128), th(2))
        for pt in pts:
            cv2.circle(frame, pt, px(4), (255,255,255), -1)

        # Risk bar
        color = danger['color']
        conf  = danger['combined_confidence']
        level = danger['danger_level']
        bar_w = int(W * 0.25)
        cv2.rectangle(frame, (px(10),px(10)), (px(10)+bar_w,px(22)), (50,50,50), -1)
        cv2.rectangle(frame, (px(10),px(10)), (px(10)+int(bar_w*conf),px(22)), color, -1)
        cv2.putText(frame, f"RISK {conf:.0%} [{level}]", (px(10),px(42)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs(0.6), color, th(2))
        cv2.putText(frame, f"Agent: {action.upper()}", (px(10),px(90)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs(0.55), (255,255,0), th(2))

        if level == "CRITICAL":
            cv2.rectangle(frame, (0,H-px(80)),(W,H-px(45)), color, -1)
            reasons = [k.replace('_',' ').upper() for k,v in danger['flags'].items() if v]
            cv2.putText(frame, f"!! {' + '.join(reasons) or 'FALL DETECTED'}",
                        (px(12),H-px(55)), cv2.FONT_HERSHEY_DUPLEX, fs(0.75), (255,255,255), th(2))

        # Timestamp + filename top right
        ts = datetime.datetime.now().strftime("%d %b %Y  %H:%M:%S")
        cv2.putText(frame, ts, (W-px(230), px(20)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs(0.5), (200,200,200), th(1))
        cv2.putText(frame, os.path.basename(video_path), (W-px(230), px(40)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs(0.45), (150,150,150), th(1))

        # MDP box — top right below timestamp
        mdp_state_val = agent.mdp.state.value if hasattr(agent, "mdp") else "N/A"
        mdp_colors = {
            "NORMAL":      (0, 200, 80),
            "LOW_RISK":    (0, 220, 200),
            "MEDIUM_RISK": (0, 165, 255),
            "HIGH_RISK":   (0, 80, 255),
            "CRITICAL":    (0, 40, 220),
        }
        mdp_col = mdp_colors.get(mdp_state_val, (200,200,200))
        if hasattr(agent, "mdp") and agent.mdp.transition_log:
            last_scores = agent.mdp.transition_log[-1].get("scores", {})
            score_txt = (f"fl={last_scores.get('floor',0):.2f} "
                         f"du={last_scores.get('duration',0):.2f} "
                         f"cf={last_scores.get('conf',0):.2f}")
        else:
            score_txt = "fl=0.00 du=0.00 cf=0.00"
        cv2.rectangle(frame, (W-px(224), px(55)),(W-px(2), px(102)), (20,20,20), -1)
        cv2.rectangle(frame, (W-px(224), px(55)),(W-px(2), px(102)), mdp_col, th(2))
        cv2.putText(frame, f"MDP: {mdp_state_val}",
                    (W-px(218), px(78)), cv2.FONT_HERSHEY_DUPLEX, fs(0.52), mdp_col, th(2))
        cv2.putText(frame, score_txt,
                    (W-px(218), px(96)), cv2.FONT_HERSHEY_SIMPLEX, fs(0.36), (180,180,180), th(1))

        # Telegram flash — top left below Agent
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
            cv2.rectangle(frame, (px(8), px(108)),(px(185), px(135)), (20,20,20), -1)
            cv2.rectangle(frame, (px(8), px(108)),(px(185), px(135)), mdp_col, th(2))
            cv2.putText(frame, flash_msg,
                        (px(14), px(128)), cv2.FONT_HERSHEY_SIMPLEX, fs(0.48), mdp_col, th(2))

        # Progress bar
        progress = frame_num / total_frames if total_frames > 0 else 0
        cv2.rectangle(frame, (0,H-px(4)),(W,H), (30,30,30), -1)
        cv2.rectangle(frame, (0,H-px(4)),(int(W*progress),H), (0,229,204), -1)
        cv2.putText(frame, f"{video_time:.1f}s / {total_frames/self.fps:.1f}s",
                    (W-px(150), H-px(8)), cv2.FONT_HERSHEY_SIMPLEX, fs(0.45), (200,200,200), th(1))

        return frame

    def process(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Skipping {video_path} — not found")
            return

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = fps
        self.last_movement_time = 0
        self.last_features      = None
        self.critical_frames    = 0
        self.low_frames         = 0
        self.feature_history.clear()
        self._init_pose()  # reinit pose for each video

        # Output video writer
        out_path = video_path.replace('.MP4', '_ARGUS.mp4').replace('.mp4', '_ARGUS.mp4')
        fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
        writer   = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

        agent     = DecisionAgent()
        frame_num = 0

        print(f"\n[ARGUS] Processing {os.path.basename(video_path)} → {os.path.basename(out_path)}")
        print(f"        {fps:.0f}fps | {W}x{H} | {total} frames | {total/fps:.1f}s")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_num += 1
            video_time = frame_num / fps

            # Stop video5 at 25s
            if os.path.basename(video_path) == "video5.MP4" and video_time >= 25:
                break

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

            action = "wait"
            danger = {'danger_level':'LOW','combined_confidence':0,
                      'fall_probability':0,'color':(0,220,80),'flags':{}}

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                features  = self.extract_features(landmarks, frame.shape)

                if features:
                    flags, immob = self.detect_events(features, video_time)
                    cv_risk      = self.compute_cv_risk(flags)
                    risk_output  = {'features': asdict(features),
                                    'flags': flags, 'risk_score': cv_risk}
                    danger       = predict_danger(risk_output)
                    danger['features'] = asdict(features)

                    if danger['danger_level'] == "CRITICAL":
                        self.critical_frames += 1
                        self.low_frames       = 0
                    else:
                        self.low_frames += 1
                        if self.low_frames > 30:
                            self.critical_frames = 0

                    danger_for_agent = dict(danger)
                    if self.critical_frames < self.SUSTAINED_FRAMES:
                        danger_for_agent['danger_level'] = "LOW"
                        danger_for_agent['combined_confidence'] *= 0.3

                    action = agent.update(danger_for_agent, video_time=video_time)
                    self.feature_history.append(features)
                    self.last_features = features
                    frame = self.draw(frame, landmarks, danger, features,
                                      action, video_time, video_path,
                                      agent, total, frame_num)
                else:
                    cv2.putText(frame, "Body not fully in frame", (10,40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,165,255), 2)

            writer.write(frame)

            # Show progress
            if frame_num % 30 == 0:
                print(f"  {video_time:.1f}s — {action} — MDP: {agent.mdp.state.value}")

        cap.release()
        writer.release()
        self.pose.close()
        print(f"[ARGUS] Saved → {out_path}")


if __name__ == "__main__":
    processor = VideoProcessor()
    base = os.path.expanduser("~/documents/synapxe!")
    for v in VIDEOS:
        path = os.path.join(base, v)
        if os.path.exists(path):
            processor.process(path)
        else:
            print(f"[SKIP] {v} not found")
    print("\n[ARGUS] All videos processed!")
