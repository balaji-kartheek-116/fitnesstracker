import streamlit as st
import importlib.util

# Define the username and password
USERNAME = "admin"
PASSWORD = "admin"

# Streamlit UI
st.title("Fitness Tracker")

# Function to run an exercise trainer script
def run_exercise_trainer(script_name):
    # Load and execute the specified trainer script
    spec = importlib.util.spec_from_file_location(script_name, f"{script_name}.py")
    trainer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer_module)

# Function to display the login form
def login():
    st.session_state["authenticated"] = False
    st.write("### Login")
    username = st.text_input("Username", "")
    password = st.text_input("Password", "", type="password")
    
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state["authenticated"] = True
            st.success("Logged in successfully!")
        else:
            st.error("Invalid username or password.")

# Function to display the logout button
def logout():
    st.session_state["authenticated"] = False
    st.success("Logged out successfully!")

# Check if the user is authenticated
if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    login()
else:
    # Show logout button
    if st.button("Logout"):
        logout()
    
    # Sidebar for exercise selection
    exercise = st.sidebar.selectbox(
        "Select Exercise",
        ["Bicep Curls", "Squats", "Push-ups", "Plank", "WorkOuts Summary"]
    )
    
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
