import streamlit as st
import cv2
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
import av
import numpy as np
import time
import os
import warnings
from datetime import datetime
import pandas as pd
from openpyxl import load_workbook

# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress google_crc32c warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="As the c extension couldn't be imported")

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

class ExerciseCounter:
    def __init__(self):
        self.stage = None
        self.count = 0

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        
        # Initialize counters for each exercise
        self.squat_counter = ExerciseCounter()
        self.pushup_counter = ExerciseCounter()
        self.bicep_curl_counter = ExerciseCounter()
        
        # Timer for planks
        self.plank_start_time = None
        self.plank_time = 0
        self.is_planking = False
        self.plank_accumulated_time = 0

        # Calories burned per rep/time
        self.calories_per_rep = {
            "squat": 0.32,
            "pushup": 0.29,
            "bicep_curl": 0.21
        }
        self.calories_per_second = 0.05  # For planks
        self.total_calories = 0
        
        # Body detection status
        self.body_fully_detected = False
        self.initial_check_done = False

    def calculate_calories(self):
        # Calculate total calories burned
        self.total_calories = (
            self.squat_counter.count * self.calories_per_rep["squat"] +
            self.pushup_counter.count * self.calories_per_rep["pushup"] +
            self.bicep_curl_counter.count * self.calories_per_rep["bicep_curl"] +
            self.plank_accumulated_time * self.calories_per_second
        )

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
            landmarks = results.pose_landmarks.landmark

            def get_coords(idx):
                return [landmarks[idx].x, landmarks[idx].y]

            try:
                # Get coordinates
                points = {
                    "shoulder_l": get_coords(11),
                    "elbow_l": get_coords(13),
                    "wrist_l": get_coords(15),
                    "hip_l": get_coords(23),
                    "knee_l": get_coords(25),
                    "ankle_l": get_coords(27),
                    "shoulder_r": get_coords(12),
                    "elbow_r": get_coords(14),
                    "wrist_r": get_coords(16),
                    "hip_r": get_coords(24),
                    "knee_r": get_coords(26),
                    "ankle_r": get_coords(28),
                }

                # Count visible landmarks
                left_points = sum(1 for key in ["shoulder_l", "elbow_l", "wrist_l", "hip_l", "knee_l", "ankle_l"] if key in points)
                right_points = sum(1 for key in ["shoulder_r", "elbow_r", "wrist_r", "hip_r", "knee_r", "ankle_r"] if key in points)
                total_points = len(landmarks)

                # Check if all 33 points are visible initially
                if not self.initial_check_done:
                    if total_points == 33:
                        self.body_fully_detected = True
                        self.initial_check_done = True
                    else:
                        raise IndexError

                # Ensure at least 5 left or 5 right points, or at least 8 points total
                if not (left_points >= 5 or right_points >= 5 or (left_points + right_points >= 8)):
                    raise IndexError

                # Calculate angles
                angle_l_knee = calculate_angle(points["hip_l"], points["knee_l"], points["ankle_l"])
                angle_r_knee = calculate_angle(points["hip_r"], points["knee_r"], points["ankle_r"])
                angle_l_elbow = calculate_angle(points["shoulder_l"], points["elbow_l"], points["wrist_l"])
                angle_r_elbow = calculate_angle(points["shoulder_r"], points["elbow_r"], points["wrist_r"])
                angle_l_hip = calculate_angle(points["shoulder_l"], points["hip_l"], points["knee_l"])
                angle_r_hip = calculate_angle(points["shoulder_r"], points["hip_r"], points["knee_r"])

                if self.body_fully_detected:
                    # Squats logic
                    if angle_l_knee > 160 and angle_r_knee > 160:
                        self.squat_counter.stage = "up"
                    if (angle_l_knee < 90 and angle_r_knee < 90) and self.squat_counter.stage == "up":
                        self.squat_counter.stage = "down"
                        self.squat_counter.count += 1
                        # Reset counters for other exercises to prevent simultaneous counting
                        self.pushup_counter.stage = None
                        self.bicep_curl_counter.stage = None

                    # Push-ups logic
                    if (angle_l_elbow > 160 and angle_r_elbow > 160 and angle_l_hip > 160 and angle_r_hip > 160 and angle_l_knee > 160 and angle_r_knee > 160):
                        self.pushup_counter.stage = "up"
                    elif (angle_l_elbow < 90 and angle_r_elbow < 90 and angle_l_hip < 180 and angle_r_hip < 180 and angle_l_knee < 180 and angle_r_knee < 180) and self.pushup_counter.stage == "up":
                        self.pushup_counter.stage = "down"
                        self.pushup_counter.count += 1
                        # Reset counters for other exercises to prevent simultaneous counting
                        self.squat_counter.stage = None
                        self.bicep_curl_counter.stage = None

                    # Bicep curls logic
                    if abs(distance_between_shoulders - distance_shoulder_to_elbow_l) < 1 and abs(distance_between_shoulders - distance_shoulder_to_elbow_r) < 1:
                        if angle_l_elbow < 30 and angle_r_elbow < 30:
                            self.bicep_curl_counter.stage = "down"
                        if (angle_l_elbow > 160 and angle_r_elbow > 160) and self.bicep_curl_counter.stage == "down":
                            self.bicep_curl_counter.stage = "up"
                            self.bicep_curl_counter.count += 1
                            # Reset counters for other exercises to prevent simultaneous counting
                            self.squat_counter.stage = None
                            self.pushup_counter.stage = None

                    # Plank logic (ensure the body is generally horizontal)
                    if (
                        (160 <= angle_l_knee <= 200 and 160 <= angle_r_knee <= 200) and 
                        (160 <= angle_l_hip <= 200 and 160 <= angle_r_hip <= 200) and
                        (0 <= angle_l_elbow <= 100 and 0 <= angle_r_elbow <= 100) and
                        (160 <= angle_l_hip <= 180 and 160 <= angle_r_hip <= 180)
                    ):
                        if not self.is_planking:
                            self.plank_start_time = time.time()
                            self.is_planking = True
                        else:
                            self.plank_time = time.time() - self.plank_start_time
                            if self.plank_time > 1:
                                self.plank_accumulated_time += self.plank_time
                                self.is_planking = False
                    else:
                        self.is_planking = False

                    # Calculate total calories
                    self.calculate_calories()

                    # Display counts and calories
                    cv2.putText(image, f"Squats: {self.squat_counter.count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Push-ups: {self.pushup_counter.count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Bicep Curls: {self.bicep_curl_counter.count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Plank Time: {self.plank_accumulated_time:.2f}s", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Total Calories Burned: {self.total_calories:.2f}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            except IndexError:
                self.body_fully_detected = False
                cv2.putText(image, 'Your body is not clearly visible', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        return av.VideoFrame.from_ndarray(image, format="bgr24")


st.write("Detects and counts squats, push-ups, bicep curls, and plank time, and calculates calories burned.")

ctx = webrtc_streamer(key="example", 
                      mode=WebRtcMode.SENDRECV, 
                      video_transformer_factory=VideoTransformer,
                      media_stream_constraints={"video": True, "audio": False}
                      )

if st.button("Show Summary and Save Data"):
    if ctx.video_transformer:
        squat_count = ctx.video_transformer.squat_counter.count
        pushup_count = ctx.video_transformer.pushup_counter.count
        bicep_curl_count = ctx.video_transformer.bicep_curl_counter.count
        plank_time = round(ctx.video_transformer.plank_accumulated_time, 2)
        total_calories = round(ctx.video_transformer.total_calories, 2)

        summary_text = f"""
        Exercise Summary:
        - Squats: {squat_count}
        - Push-ups: {pushup_count}
        - Bicep Curls: {bicep_curl_count}
        - Plank Time: {plank_time} seconds
        - Total Calories Burned: {total_calories} calories
        """
        
        st.text(summary_text)

        # Create a paragraph summary
        summary_paragraph = f"Today, you completed {squat_count} squats, {pushup_count} push-ups, {bicep_curl_count} bicep curls, and held a plank for {plank_time} seconds, burning a total of {total_calories} calories. Great job on your workout!"

        st.write(summary_paragraph)

        # Save to Excel
        data_file = "exercise_data.xlsx"
        date_today = datetime.now().strftime("%Y-%m-%d")
        time_now = datetime.now().strftime("%H:%M:%S")
        
        if os.path.exists(data_file):
            df = pd.read_excel(data_file, sheet_name='Sheet1')
        else:
            df = pd.DataFrame(columns=["Date", "Time", "Squats", "Push-ups", "Bicep-Curls", "Plank time", "Total Calories Burned"])
        
        new_data = {
            "Date": date_today,
            "Time": time_now,
            "Squats": squat_count,
            "Push-ups": pushup_count,
            "Bicep-Curls": bicep_curl_count,
            "Plank time": plank_time,
            "Total Calories Burned": total_calories
        }

        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        
        # Save to the same sheet without errors
        with pd.ExcelWriter(data_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')

        st.success("Summary saved to Excel!")