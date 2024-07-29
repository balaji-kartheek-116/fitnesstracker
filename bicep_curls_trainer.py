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

# Function to check bicep curl form and provide feedback
def check_bicep_curl_form(points):
    feedback = []

    # Calculate angles
    angle_l_elbow = calculate_angle(points["shoulder_l"], points["elbow_l"], points["wrist_l"])
    angle_r_elbow = calculate_angle(points["shoulder_r"], points["elbow_r"], points["wrist_r"])

    # Rule 1: Elbow Position
    elbow_position = angle_l_elbow < 30 or angle_r_elbow < 30

    # Rule 2: Full Range of Motion
    full_range = angle_l_elbow > 160 or angle_r_elbow > 160

    if not elbow_position:
        feedback.append("Fully bend your elbows")
    elif not full_range:
        feedback.append("Extend your arms fully")
    else:
        feedback.append("You are doing bicep curls correctly")

    return feedback

class BicepCurlTransformer(VideoTransformerBase):
    def __init__(self):
        self.bicep_curl_counter = 0
        self.stage = None
        self.start_time = time.time()
        self.messages = [
            "Instructions: Stand with feet shoulder-width apart,Keep elbows close to your torso,Fully extend your arms,Curl the weights up"
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
                "shoulder_r": get_coords(landmarks, 12),
                "elbow_r": get_coords(landmarks, 14),
                "wrist_r": get_coords(landmarks, 16),
            }

            # Check bicep curl form
            feedback = check_bicep_curl_form(points)

            # Calculate angle for bicep curl detection
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

            # Detect bicep curls
            if angle_l_elbow > 160 and angle_r_elbow > 160:
                self.stage = "down"
            elif angle_l_elbow < 30 and angle_r_elbow < 30 and self.stage == "down":
                self.stage = "up"
                self.bicep_curl_counter += 1

            # Update instructional message every 5 seconds
            current_time = time.time()
            if current_time - self.start_time >= 5:
                self.message_index = (self.message_index + 1) % len(self.messages)
                self.start_time = current_time

            # Display bicep curl count and feedback
            cv2.putText(image, self.messages[self.message_index], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            y = 60
            for line in feedback:
                cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                y += 30
            cv2.putText(image, f"Bicep Curls: {self.bicep_curl_counter}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            cv2.putText(image, "Do a valid bicep curl", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Streamlit UI
st.title("Bicep Curl Form Checker")

run = st.checkbox('Start Webcam')

if run:
    webrtc_streamer(
        key="bicep_curl",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=BicepCurlTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )
else:
    st.write("Click the checkbox to start the webcam")
