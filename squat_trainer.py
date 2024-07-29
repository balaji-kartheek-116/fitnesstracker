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

# Function to check squat form and provide feedback
def check_squat_form(points):
    feedback = []

    # Calculate angles
    angle_l_knee = calculate_angle(points["hip_l"], points["knee_l"], points["ankle_l"])
    angle_r_knee = calculate_angle(points["hip_r"], points["knee_r"], points["ankle_r"])

    # Rule 1: Knee Position
    knee_bent = angle_l_knee < 90 and angle_r_knee < 90

    # Rule 2: Hip Position
    hip_straight = angle_l_knee > 160 and angle_r_knee > 160

    if not knee_bent:
        feedback.append("Bend your knees more")
    elif not hip_straight:
        feedback.append("Keep your hips aligned with your knees")
    else:
        feedback.append("You are doing squats correctly")

    return feedback

class SquatTransformer(VideoTransformerBase):
    def __init__(self):
        self.squat_counter = 0
        self.stage = None
        self.start_time = time.time()
        self.messages = [
            " Instructions: Feet shoulder-width apart,Lower your body until thighs are parallel to the ground, Keep your back straight, Push through your heels to stand up"
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
                "hip_l": get_coords(landmarks, 23),
                "knee_l": get_coords(landmarks, 25),
                "ankle_l": get_coords(landmarks, 27),
                "hip_r": get_coords(landmarks, 24),
                "knee_r": get_coords(landmarks, 26),
                "ankle_r": get_coords(landmarks, 28),
            }

            # Check squat form
            feedback = check_squat_form(points)

            # Calculate angle for squat detection
            angle_l_knee = calculate_angle(
                points["hip_l"],
                points["knee_l"],
                points["ankle_l"]
            )
            angle_r_knee = calculate_angle(
                points["hip_r"],
                points["knee_r"],
                points["ankle_r"]
            )

            # Detect squats
            if angle_l_knee > 160 and angle_r_knee > 160:
                self.stage = "up"
            elif angle_l_knee < 90 and angle_r_knee < 90 and self.stage == "up":
                self.stage = "down"
                self.squat_counter += 1

            # Update instructional message every 5 seconds
            current_time = time.time()
            if current_time - self.start_time >= 5:
                self.message_index = (self.message_index + 1) % len(self.messages)
                self.start_time = current_time

            # Display squat count and feedback
            cv2.putText(image, self.messages[self.message_index], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            y = 60
            for line in feedback:
                cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                y += 30
            cv2.putText(image, f"Squats: {self.squat_counter}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            cv2.putText(image, "Do a valid squat", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Streamlit UI
st.title("Squat Form Checker")

run = st.checkbox('Start Webcam')

if run:
    webrtc_streamer(
        key="squat",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=SquatTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )
else:
    st.write("Click the checkbox to start the webcam")
