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

# Function to check push-up form and provide feedback
def check_pushup_form(points):
    feedback = []

    # Calculate angles
    angle_l_elbow = calculate_angle(points["shoulder_l"], points["elbow_l"], points["wrist_l"])
    angle_r_elbow = calculate_angle(points["shoulder_r"], points["elbow_r"], points["wrist_r"])
    angle_l_hip = calculate_angle(points["shoulder_l"], points["hip_l"], points["ankle_l"])
    angle_r_hip = calculate_angle(points["shoulder_r"], points["hip_r"], points["ankle_r"])
    angle_l_knee = calculate_angle(points["hip_l"], points["knee_l"], points["ankle_l"])
    angle_r_knee = calculate_angle(points["hip_r"], points["knee_r"], points["ankle_r"])

    # Rule 1: Elbow Position (during lowering phase)
    elbow_position = angle_l_elbow < 180 and angle_r_elbow < 180  # Slightly relaxed angle

    # Rule 2: Hip Position
    hip_straight = (150 <= angle_l_hip <= 210) and (150 <= angle_r_hip <= 210)

    # Rule 3: Leg Position
    leg_straight = (150 <= angle_l_knee <= 210) and (150 <= angle_r_knee <= 210)

    if not elbow_position:
        feedback.append("Close elbows to 90 degrees and open it to 180 degrees ")
    elif not hip_straight:
        feedback.append("Keep the hips aligned with shoulders and ankles")
    elif not leg_straight:
        feedback.append("Keep your legs straight")
    else:
        feedback.append("You are doing push-up correctly")

    return feedback

class PushUpTransformer(VideoTransformerBase):
    def __init__(self):
        self.pushup_counter = 0
        self.stage = None
        self.pushup_start_time = None
        self.start_time = time.time()
        self.messages = [
            "Face down on ground"
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
                "elbow_l": get_coords(landmarks, 13),
                "wrist_l": get_coords(landmarks, 15),
                "hip_l": get_coords(landmarks, 23),
                "knee_l": get_coords(landmarks, 25),
                "ankle_l": get_coords(landmarks, 27),
                "shoulder_r": get_coords(landmarks, 12),
                "elbow_r": get_coords(landmarks, 14),
                "wrist_r": get_coords(landmarks, 16),
                "hip_r": get_coords(landmarks, 24),
                "knee_r": get_coords(landmarks, 26),
                "ankle_r": get_coords(landmarks, 28),
            }

            # Check push-up form
            feedback = check_pushup_form(points)

            # Calculate angle for push-up detection
            angle_l_elbow = calculate_angle(
                points["shoulder_l"],
                points["elbow_l"],
                points["wrist_l"]
            )
            angle_r_elbow = calculate_angle(
                points["shoulder_r"],
                points["elbow_r"],
                points["wrist_r"]
            )

            # Detect push-up reps
            if angle_l_elbow > 160 and angle_r_elbow > 160:
                self.stage = "up"
            elif angle_l_elbow < 110 and angle_r_elbow < 110 and self.stage == "up":
                self.stage = "down"
                self.pushup_counter += 1

            # Update instructional message every 5 seconds
            current_time = time.time()
            if current_time - self.start_time >= 5:
                self.message_index = (self.message_index + 1) % len(self.messages)
                self.start_time = current_time

            # Display push-up count and feedback
            cv2.putText(image, self.messages[self.message_index], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            y = 60
            for line in feedback:
                cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                y += 30
            cv2.putText(image, f"Push-ups: {self.pushup_counter}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            cv2.putText(image, "Do a valid push-up", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Streamlit UI
st.title("Push-Up Form Checker")

run = st.checkbox('Start Webcam')

if run:
    webrtc_streamer(
        key="pushup",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PushUpTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )
else:
    st.write("Click the checkbox to start the webcam")
