"""
Activity Monitor — learns patient's daily routine and flags anomalies.
Runs as a background service alongside the main fall detector.
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import json
import time
import os
import threading
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from collections import defaultdict

TELEGRAM_TOKEN   = "8676872825:AAGTT3ZzEfGtAeeeaZZcTLdRT9_Zuil9C8c"
TELEGRAM_CHAT_ID = "5740866477"

# Config
ACTIVITY_LOG_FILE   = "activity_log.jsonl"
BASELINE_FILE       = "activity_baseline.json"
MIN_DAYS_TO_LEARN   = 2       # days before anomaly detection kicks in
ANOMALY_THRESHOLD   = 2.0     # std deviations below normal = anomaly
INACTIVITY_ALERT_MINS = 30    # mins of unexpected inactivity before alerting
SAMPLE_INTERVAL_SECS  = 30    # record activity every 30 seconds
ALERT_COOLDOWN_MINS   = 60    # min time between welfare check alerts
MODEL_PATH = "pose_landmarker.task"

def send_telegram(message):
    try:
        url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }).encode()
        urllib.request.urlopen(url, data, timeout=5)
        print(f"[ACTIVITY] Telegram sent")
    except Exception as e:
        print(f"[ACTIVITY TELEGRAM ERROR] {e}")

def send_async(msg):
    threading.Thread(target=send_telegram, args=(msg,), daemon=True).start()


class ActivityMonitor:
    def __init__(self):
        # MediaPipe for motion detection
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self.pose = mp_pose = mp_vision.PoseLandmarker.create_from_options(options)

        self.baseline          = self._load_baseline()
        self.last_positions    = {}
        self.inactivity_start  = None
        self.last_alert_time   = 0
        self.consecutive_inactive = 0

        print("[ACTIVITY] Monitor initialized")
        print(f"[ACTIVITY] Baseline days collected: {self._days_collected()}")

    def _load_baseline(self) -> dict:
        if os.path.exists(BASELINE_FILE):
            with open(BASELINE_FILE) as f:
                return json.load(f)
        return {}

    def _save_baseline(self):
        with open(BASELINE_FILE, 'w') as f:
            json.dump(self.baseline, f, indent=2)

    def _days_collected(self) -> int:
        if not os.path.exists(ACTIVITY_LOG_FILE):
            return 0
        dates = set()
        with open(ACTIVITY_LOG_FILE) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    dates.add(entry['date'])
                except:
                    pass
        return len(dates)

    def measure_activity(self, frame) -> float:
        """
        Returns activity score 0-1 for current frame.
        0 = no person / no movement, 1 = high movement.
        """
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )
        if not results.pose_landmarks:
            return 0.0

        lm = results.pose_landmarks[0]
        # Track movement of key points vs last frame
        key_points = [0, 11, 12, 23, 24]  # nose, shoulders, hips
        current_pos = {i: (lm[i].x, lm[i].y) for i in key_points
                      if lm[i].visibility > 0.4}

        if not current_pos:
            return 0.0

        # Compute movement vs last sample
        movement = 0.0
        if self.last_positions:
            shared = set(current_pos.keys()) & set(self.last_positions.keys())
            if shared:
                diffs = [abs(current_pos[i][0] - self.last_positions[i][0]) +
                         abs(current_pos[i][1] - self.last_positions[i][1])
                         for i in shared]
                movement = min(np.mean(diffs) * 20, 1.0)  # normalize to 0-1

        self.last_positions = current_pos
        return float(movement)

    def log_activity(self, activity_score: float):
        """Log activity sample with timestamp."""
        now  = datetime.now()
        entry = {
            'timestamp': time.time(),
            'date':      now.strftime('%Y-%m-%d'),
            'hour':      now.hour,
            'minute':    now.minute,
            'activity':  round(activity_score, 4),
            'person_present': activity_score > 0
        }
        with open(ACTIVITY_LOG_FILE, 'a') as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def update_baseline(self):
        """
        Rebuild hourly baseline from all logged data.
        For each hour: mean and std of activity scores.
        """
        if not os.path.exists(ACTIVITY_LOG_FILE):
            return

        hourly = defaultdict(list)
        with open(ACTIVITY_LOG_FILE) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    hourly[e['hour']].append(e['activity'])
                except:
                    pass

        baseline = {}
        for hour, scores in hourly.items():
            if len(scores) >= 5:
                baseline[str(hour)] = {
                    'mean': round(float(np.mean(scores)), 4),
                    'std':  round(float(np.std(scores)), 4),
                    'samples': len(scores)
                }
        self.baseline = baseline
        self._save_baseline()
        print(f"[ACTIVITY] Baseline updated — {len(baseline)} hours profiled")

    def check_anomaly(self, activity_score: float) -> dict:
        """
        Compare current activity to baseline for this hour.
        Returns anomaly info if detected.
        """
        hour     = str(datetime.now().hour)
        now_time = datetime.now().strftime('%H:%M')

        if self._days_collected() < MIN_DAYS_TO_LEARN:
            return {'anomaly': False, 'reason': 'still_learning'}

        if hour not in self.baseline:
            return {'anomaly': False, 'reason': 'no_baseline_for_hour'}

        b    = self.baseline[hour]
        mean = b['mean']
        std  = b['std'] if b['std'] > 0.01 else 0.01

        z_score = (activity_score - mean) / std

        # Unexpectedly inactive (person normally active at this hour)
        if z_score < -ANOMALY_THRESHOLD and mean > 0.05:
            return {
                'anomaly': True,
                'type': 'unexpected_inactivity',
                'expected_activity': round(mean, 3),
                'actual_activity': round(activity_score, 3),
                'z_score': round(z_score, 2),
                'hour': hour,
                'time': now_time
            }

        return {'anomaly': False, 'z_score': round(z_score, 2)}

    def run(self):
        """Main monitoring loop — runs 24/7."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("[ACTIVITY] Monitoring started")
        last_sample_time    = 0
        last_baseline_update = 0
        inactivity_samples  = 0
        last_welfare_alert  = 0

        try:
            profile = json.load(open('patient_profile.json'))
            patient_name = profile.get('name', 'Patient')
        except:
            patient_name = 'Patient'

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                continue

            now = time.time()

            # Sample every SAMPLE_INTERVAL_SECS
            if now - last_sample_time >= SAMPLE_INTERVAL_SECS:
                last_sample_time = now
                activity = self.measure_activity(frame)
                entry    = self.log_activity(activity)
                anomaly  = self.check_anomaly(activity)

                hour = datetime.now().hour
                print(f"[ACTIVITY] {datetime.now().strftime('%H:%M')} — "
                      f"activity={activity:.3f} | "
                      f"anomaly={anomaly.get('anomaly')} | "
                      f"z={anomaly.get('z_score', 'N/A')}")

                # Track consecutive inactive samples
                if activity < 0.02:
                    inactivity_samples += 1
                else:
                    inactivity_samples = 0

                # Welfare check — unexpected inactivity for sustained period
                inactive_mins = (inactivity_samples * SAMPLE_INTERVAL_SECS) / 60
                cooldown_ok   = (now - last_welfare_alert) > (ALERT_COOLDOWN_MINS * 60)

                if (anomaly.get('anomaly') and
                        inactive_mins >= INACTIVITY_ALERT_MINS and
                        cooldown_ok):

                    b = self.baseline.get(str(hour), {})
                    send_async(
                        f"⚠️ <b>Welfare Check — ARGUS</b>\n\n"
                        f"👤 <b>{patient_name}</b> has been inactive for "
                        f"<b>{inactive_mins:.0f} minutes</b> at "
                        f"{datetime.now().strftime('%H:%M')}.\n\n"
                        f"📊 Expected activity at this hour: "
                        f"{b.get('mean', 0):.0%}\n"
                        f"📊 Actual activity: {activity:.0%}\n\n"
                        f"This is unusual for their normal routine.\n"
                        f"Please check in when possible. 🙏"
                    )
                    last_welfare_alert = now
                    print(f"[ACTIVITY] Welfare check sent — {inactive_mins:.0f} mins inactive")

                # Update baseline every hour
                if now - last_baseline_update > 3600:
                    last_baseline_update = now
                    threading.Thread(target=self.update_baseline, daemon=True).start()

            # Show simple activity feed
            activity_display = self.measure_activity(frame)
            H, W = frame.shape[:2]
            bar  = int(W * 0.3 * activity_display)
            cv2.rectangle(frame, (10,10), (10+int(W*0.3), 25), (50,50,50), -1)
            cv2.rectangle(frame, (10,10), (10+bar, 25), (0,220,80), -1)
            cv2.putText(frame, f"Activity: {activity_display:.2f}", (10,45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            cv2.putText(frame, f"Days learned: {self._days_collected()}/{MIN_DAYS_TO_LEARN}",
                        (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(frame, f"Inactive samples: {inactivity_samples}",
                        (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.imshow("ARGUS — Activity Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.pose.close()

    def release(self):
        pass


if __name__ == "__main__":
    monitor = ActivityMonitor()

    # If enough data, show baseline summary
    if monitor._days_collected() >= MIN_DAYS_TO_LEARN:
        print("\nCurrent baseline:")
        for hour in sorted(monitor.baseline.keys(), key=int):
            b = monitor.baseline[hour]
            print(f"  {int(hour):02d}:00 — mean={b['mean']:.3f} std={b['std']:.3f} ({b['samples']} samples)")
    else:
        days = monitor._days_collected()
        print(f"\nStill learning... {days}/{MIN_DAYS_TO_LEARN} days collected")
        print("Run this monitor for a few days to build the baseline.")

    print("\nStarting activity monitor — press Q to quit")
    monitor.run()
