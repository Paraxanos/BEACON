import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import json
import hashlib
import sqlite3
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

# --------------------------
# Configuration & Constants
# --------------------------
ANOMALY_THRESHOLD = 0.65
INITIAL_PROFILING_ACTIONS = 10
TRUST_SCORE_RANGE = (0, 100)
SESSION_TIMEOUT_MINUTES = 30

# --------------------------
# Database Setup
# --------------------------
def init_db():
    conn = sqlite3.connect('beacon.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 username TEXT UNIQUE,
                 password_hash TEXT,
                 created_at TIMESTAMP,
                 last_login TIMESTAMP,
                 baseline_profile TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS behavior_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 user_id INTEGER,
                 timestamp TIMESTAMP,
                 event_type TEXT,
                 behavior_data TEXT,
                 anomaly_score REAL,
                 processed BOOLEAN)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS trust_scores
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 user_id INTEGER,
                 timestamp TIMESTAMP,
                 score REAL,
                 reason TEXT)''')
    
    conn.commit()
    conn.close()

init_db()

# --------------------------
# Helper Functions
# --------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_hash, password):
    return stored_hash == hash_password(password)

def get_user(username):
    conn = sqlite3.connect('beacon.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    return user

def create_user(username, password):
    conn = sqlite3.connect('beacon.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                  (username, hash_password(password), datetime.now()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def log_behavior(user_id, event_type, behavior_data, anomaly_score=None):
    conn = sqlite3.connect('beacon.db')
    c = conn.cursor()
    c.execute("INSERT INTO behavior_logs (user_id, timestamp, event_type, behavior_data, anomaly_score, processed) VALUES (?, ?, ?, ?, ?, ?)",
              (user_id, datetime.now(), event_type, json.dumps(behavior_data), anomaly_score, False))
    conn.commit()
    conn.close()

def update_trust_score(user_id, score, reason):
    conn = sqlite3.connect('beacon.db')
    c = conn.cursor()
    c.execute("INSERT INTO trust_scores (user_id, timestamp, score, reason) VALUES (?, ?, ?, ?)",
              (user_id, datetime.now(), score, reason))
    conn.commit()
    conn.close()

def get_latest_trust_score(user_id):
    conn = sqlite3.connect('beacon.db')
    c = conn.cursor()
    c.execute("SELECT score FROM trust_scores WHERE user_id=? ORDER BY timestamp DESC LIMIT 1", (user_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else 80  # Default starting score

# --------------------------
# Behavioral Models
# --------------------------
class BehaviorAnalyzer:
    def __init__(self):
        self.models = {
            'typing_rhythm': IsolationForest(contamination=0.1),
            'mouse_movements': IsolationForest(contamination=0.1),
            'navigation_patterns': IsolationForest(contamination=0.1)
        }
        self.baseline_profiles = {}
    
    def train_for_user(self, user_id, behavior_data):
        features = self._extract_features(behavior_data)
        for model_name, model in self.models.items():
            model.fit(features[model_name])
        self.baseline_profiles[user_id] = features
    
    def analyze_behavior(self, user_id, current_behavior):
        if user_id not in self.baseline_profiles:
            return 0.0
            
        features = self._extract_features([current_behavior])
        anomaly_scores = []
        
        for model_name, model in self.models.items():
            score = model.decision_function(features[model_name])[0]
            anomaly_scores.append(score)
        
        avg_score = np.mean(anomaly_scores)
        normalized_score = 1 / (1 + np.exp(-avg_score))
        return normalized_score
    
    def _extract_features(self, behavior_data):
        return {
            'typing_rhythm': np.random.rand(len(behavior_data), 3),
            'mouse_movements': np.random.rand(len(behavior_data), 4),
            'navigation_patterns': np.random.rand(len(behavior_data), 2)
        }

behavior_analyzer = BehaviorAnalyzer()

# --------------------------
# Mock Banking Functions
# --------------------------
def mock_transfer_funds(user_id, amount, recipient):
    behavior = {
        'action': 'transfer',
        'amount': amount,
        'recipient': recipient,
        'time_spent': np.random.uniform(2.0, 10.0),
        'keystroke_timing': [np.random.uniform(0.1, 0.3) for _ in range(10)],
        'mouse_movements': np.random.randint(5, 20)
    }
    
    anomaly_score = behavior_analyzer.analyze_behavior(user_id, behavior)
    log_behavior(user_id, 'transaction', behavior, anomaly_score)
    
    current_score = get_latest_trust_score(user_id)
    new_score = current_score * (1 - anomaly_score/2)
    update_trust_score(user_id, new_score, f"Transaction anomaly: {anomaly_score:.2f}")
    
    return anomaly_score < ANOMALY_THRESHOLD

# --------------------------
# UI Components
# --------------------------
def trust_score_gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Trust Score"},
        gauge = {
            'axis': {'range': TRUST_SCORE_RANGE},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "red"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score}}))
    
    st.plotly_chart(fig, use_container_width=True)

def behavior_radar_chart(user_id):
    categories = ['Typing Rhythm', 'Mouse Patterns', 'Navigation', 'Session Timing', 'Decision Speed']
    baseline = [0.8, 0.7, 0.9, 0.6, 0.75]
    current = [np.random.uniform(0.5, 1.0) for _ in range(5)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=baseline,
        theta=categories,
        fill='toself',
        name='Your Baseline'
    ))
    fig.add_trace(go.Scatterpolar(
        r=current,
        theta=categories,
        fill='toself',
        name='Current Session'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Main App Logic
# --------------------------
def login_page():
    st.title("BEACON Authentication Prototype")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                user = get_user(username)
                if user and verify_password(user[2], password):
                    st.session_state['user'] = {
                        'id': user[0],
                        'username': user[1],
                        'last_login': datetime.now()
                    }
                    st.success("Login successful!")
                    time.sleep(1)
                    st.rerun()  # Changed from experimental_rerun()
                else:
                    st.error("Invalid username or password")
    
    with tab2:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username")
            new_password = st.text_input("Choose a password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create Account")
            
            if submitted:
                if new_password != confirm_password:
                    st.error("Passwords don't match")
                elif create_user(new_username, new_password):
                    st.success("Account created! Please log in.")
                else:
                    st.error("Username already exists")

def dashboard_page():
    user_id = st.session_state.user['id']
    username = st.session_state.user['username']
    
    st.title(f"Welcome to BEACON Bank, {username}!")
    
    st.sidebar.header("Navigation")
    page_options = ["Dashboard", "Transfer Funds", "Behavior Profile", "Log Out"]
    selected_page = st.sidebar.radio("Go to", page_options)
    
    if selected_page == "Log Out":
        st.session_state.clear()
        st.rerun()  # Changed from experimental_rerun()
    
    if selected_page == "Dashboard":
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Your Trust Score")
            current_score = get_latest_trust_score(user_id)
            trust_score_gauge(current_score)
            
            if current_score < 40:
                st.error("High risk detected! Verify your identity.")
            elif current_score < 70:
                st.warning("Moderate risk detected. Be cautious.")
            else:
                st.success("Normal behavior detected.")
        
        with col2:
            st.subheader("Recent Activity")
            behavior_radar_chart(user_id)
            
            if np.random.random() > 0.7:
                st.warning("Unusual typing rhythm detected")
            if np.random.random() > 0.8:
                st.error("Suspicious mouse movement pattern")
    
    elif selected_page == "Transfer Funds":
        st.subheader("Transfer Funds")
        
        with st.form("transfer_form"):
            amount = st.number_input("Amount", min_value=1, max_value=10000)
            recipient = st.text_input("Recipient Account Number")
            memo = st.text_input("Memo (Optional)")
            
            submitted = st.form_submit_button("Transfer")
            
            if submitted:
                success = mock_transfer_funds(user_id, amount, recipient)
                if success:
                    st.success(f"Successfully transferred ${amount} to {recipient}")
                else:
                    st.error("Transaction blocked due to suspicious behavior")
    
    elif selected_page == "Behavior Profile":
        st.subheader("Your Behavioral Profile")
        
        tab1, tab2, tab3 = st.tabs(["Patterns", "History", "Settings"])
        
        with tab1:
            st.write("Your typical behavior patterns:")
            behavior_radar_chart(user_id)
            
            st.write("""
            ### What we monitor:
            - **Typing Rhythm**: How you type (speed, rhythm)
            - **Mouse Patterns**: How you move your cursor
            - **Navigation**: How you move through the app
            - **Session Timing**: When and how long you use the app
            - **Decision Speed**: How quickly you make transactions
            """)
        
        with tab2:
            st.write("Historical trust scores:")
            
            dates = pd.date_range(end=datetime.now(), periods=30).tolist()
            scores = np.clip(np.cumsum(np.random.normal(0, 5, 30)) + 80, 0, 100)
            
            fig = px.line(x=dates, y=scores, 
                         labels={'x': 'Date', 'y': 'Trust Score'},
                         title="Your Trust Score Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.write("Behavioral Authentication Settings")
            
            st.checkbox("Enable enhanced monitoring", value=True)
            st.checkbox("Receive security alerts", value=True)
            st.checkbox("Share anonymized data to improve security", value=False)
            
            if st.button("Reset Behavioral Baseline"):
                st.warning("This will require you to re-establish your behavior patterns")

# --------------------------
# App Router
# --------------------------
def main():
    if 'user' not in st.session_state:
        login_page()
    else:
        dashboard_page()

if __name__ == "__main__":
    main()