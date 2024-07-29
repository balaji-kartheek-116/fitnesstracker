import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
import av
import time

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Function to get coordinates
def get_coords(landmarks, idx):
    return [landmarks[idx].x, landmarks[idx].y]

# Function to check plank form and provide feedback
def check_plank_form(points):
    feedback = []

    # Calculate angles
    angle_l_hip = calculate_angle(points["shoulder_l"], points["hip_l"], points["ankle_l"])
    angle_r_hip = calculate_angle(points["shoulder_r"], points["hip_r"], points["ankle_r"])

    # Rule 1: Body Alignment
    body_aligned = (160 <= angle_l_hip <= 200) and (160 <= angle_r_hip <= 200)

    if not body_aligned:
        feedback.append("Keep your body in a straight line from head to heels")
    else:
        feedback.append("You are doing the plank correctly")

    return feedback

class PlankTransformer(VideoTransformerBase):
    def __init__(self):
        self.plank_start_time = None
        self.plank_duration = 0
        self.start_time = time.time()
        self.messages = [
            "Hold the plank position",
            "Keep your body in a straight line",
            "Engage your core",
            "Breathe steadily",
        ]
        self.message_index = 0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            points = {
                "shoulder_l": get_coords(landmarks, 11),
                "hip_l": get_coords(landmarks, 23),
                "ankle_l": get_coords(landmarks, 27),
                "shoulder_r": get_coords(landmarks, 12),
                "hip_r": get_coords(landmarks, 24),
                "ankle_r": get_coords(landmarks, 28),
            }

            # Check plank form
            feedback = check_plank_form(points)

            # Calculate angle for plank detection
            angle_l_hip = calculate_angle(
                points["shoulder_l"],
                points["hip_l"],
                points["ankle_l"]
            )
            angle_r_hip = calculate_angle(
                points["shoulder_r"],
                points["hip_r"],
                points["ankle_r"]
            )

            # Detect plank
            if angle_l_hip < 200 and angle_r_hip < 200:
                if self.plank_start_time is None:
                    self.plank_start_time = time.time()
                self.plank_duration = time.time() - self.plank_start_time
            else:
                self.plank_start_time = None

            # Update instructional message every 2 seconds
            current_time = time.time()
            if current_time - self.start_time >= 2:
                self.message_index = (self.message_index + 1) % len(self.messages)
                self.start_time = current_time

            # Display plank duration and feedback
            cv2.putText(image, self.messages[self.message_index], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            y = 60
            for line in feedback:
                cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                y += 30
            cv2.putText(image, f"Plank Duration: {int(self.plank_duration)} seconds", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            cv2.putText(image, "Do a valid plank", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Streamlit UI
st.title("Plank Form Checker")

run = st.checkbox('Start Webcam')

if run:
    webrtc_streamer(
        key="plank",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PlankTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )
else:
    st.write("Click the checkbox to start the webcam")
