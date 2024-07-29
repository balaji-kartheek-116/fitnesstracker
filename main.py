import streamlit as st
import importlib.util

# Streamlit UI
st.title("Fitness Tracker")

# Sidebar for exercise selection
exercise = st.sidebar.selectbox(
    "Select Exercise",
    ["Bicep Curls", "Squats", "Push-ups", "Plank", "WorkOuts Summary"]
)

# Function to run an exercise trainer script
def run_exercise_trainer(script_name):
    # Load and execute the specified trainer script
    spec = importlib.util.spec_from_file_location(script_name, f"{script_name}.py")
    trainer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer_module)

# Execute the selected exercise script
if exercise == "Push-ups":
    run_exercise_trainer("push_up_trainer")
elif exercise == "Bicep Curls":
    run_exercise_trainer("bicep_curls_trainer")
elif exercise == "Plank":
    run_exercise_trainer("plank_trainer")
elif exercise == "Squats":
    run_exercise_trainer("squat_trainer")
elif exercise == "WorkOuts Summary":
    run_exercise_trainer("exercises5")
else:
    st.write(f"Selected exercise: {exercise}. Specific trainer not implemented yet.")
