import streamlit as st
import cv2
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from collections import deque

st.set_page_config(page_title="Posture Detection", layout="centered")
st.title("🧍 AI Posture Detection (Web App)")
st.write("Allow camera access and sit straight for calibration.")

mp_pose = mp.solutions.pose

ANGLE_BUFFER = 15
neck_buffer = deque(maxlen=ANGLE_BUFFER)
torso_buffer = deque(maxlen=ANGLE_BUFFER)

def angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p1[1] - p2[1]
    return abs(math.degrees(math.atan2(dx, dy)))

class PostureProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose()
        self.calibrated = False
        self.neck_base = 0
        self.torso_base = 0
        self.frames = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            ear = lm[mp_pose.PoseLandmark.LEFT_EAR]
            hip = lm[mp_pose.PoseLandmark.LEFT_HIP]

            sh_p = (int(sh.x * w), int(sh.y * h))
            ear_p = (int(ear.x * w), int(ear.y * h))
            hip_p = (int(hip.x * w), int(hip.y * h))

            neck = angle(sh_p, ear_p)
            torso = angle(hip_p, sh_p)

            neck_buffer.append(neck)
            torso_buffer.append(torso)

            neck_avg = sum(neck_buffer) / len(neck_buffer)
            torso_avg = sum(torso_buffer) / len(torso_buffer)

            if not self.calibrated:
                self.frames += 1
                if self.frames > 100:
                    self.neck_base = neck_avg
                    self.torso_base = torso_avg
                    self.calibrated = True
                cv2.putText(img, "Calibrating...", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            else:
                neck_thresh = self.neck_base + 15
                torso_thresh = self.torso_base + 8

                if neck_avg < neck_thresh and torso_avg < torso_thresh:
                    status = "GOOD POSTURE"
                    color = (0,255,0)
                else:
                    status = "BAD POSTURE"
                    color = (0,0,255)

                cv2.putText(img, status, (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            cv2.line(img, sh_p, ear_p, (255,255,0), 3)
            cv2.line(img, hip_p, sh_p, (255,255,0), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="posture",
    video_processor_factory=PostureProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
