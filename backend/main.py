from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import datetime
import json
import os
import joblib # For loading the model
import pandas as pd # For DataFrame operations during feature engineering
import numpy as np # For numerical operations
import math # For math.sqrt in feature engineering
from sklearn.preprocessing import StandardScaler
from fastapi.middleware.cors import CORSMiddleware

# --- Configuration ---
BEHAVIOR_LOG_FILE = "behavior_logs.jsonl" # .jsonl for JSON Lines format
MODEL_PATH = "isolation_forest_model.joblib"
# We need to re-fit a StandardScaler to the training data's numerical columns
# to ensure it scales new incoming data consistently.
# We'll load the full processed CSV to get the training data for the scaler.
PROCESSED_DATA_PATH = "processed_behavior_features.csv" 

# Initialize FastAPI app
app = FastAPI(
    title="BEACON Behavioral Backend",
    description="API for receiving and processing behavioral data from the BEACON web app.",
    version="0.1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for the *inner* 'data' field's structure
class BehaviorDataPayload(BaseModel): # Renamed for clarity
    typingPatterns: Optional[List[Dict[str, Any]]] = []
    mouseMovements: Optional[List[Dict[str, Any]]] = []
    clickEvents: Optional[List[Dict[str, Any]]] = []
    deviceInfo: Optional[Dict[str, Any]] = {}
    fingerprint: Optional[str] = None

# Pydantic model for the *full incoming request body*
class BehaviorLogEntry(BaseModel): # This is the main model the API expects
    timestamp: str
    user_email: Optional[str] = None
    data: BehaviorDataPayload # Now uses the specific payload model

# --- Global Variables for ML Model and Scaler ---
# These will be loaded once when the application starts
ml_model = None
scaler = None
numerical_cols_to_scale = None # To store column names that need scaling

# --- Helper functions for file-based storage ---
def load_behavior_logs() -> List[BehaviorLogEntry]:
    """Loads behavior logs from the JSONL file."""
    logs = []
    if os.path.exists(BEHAVIOR_LOG_FILE):
        with open(BEHAVIOR_LOG_FILE, 'r') as f:
            for line in f:
                try:
                    log_data = json.loads(line.strip())
                    logs.append(BehaviorLogEntry(**log_data))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from log file: {line.strip()} - {e}")
                except Exception as e:
                    print(f"Error creating BehaviorLogEntry from log file data: {log_data} - {e}")
    return logs

def save_behavior_log(log_entry: BehaviorLogEntry):
    """Appends a new behavior log entry to the JSONL file."""
    with open(BEHAVIOR_LOG_FILE, 'a') as f: # 'a' for append mode
        f.write(json.dumps(log_entry.model_dump()) + '\n') # .model_dump() is for Pydantic v2

# --- Feature Extraction Function (Copied from Phase 2) ---
# This function must be identical to the one used during training
def extract_features(behavior_data: Dict[str, Any]):
    features = {}

    # --- 1. Typing Patterns Features ---
    typing_patterns = behavior_data.get('typingPatterns', [])
    if typing_patterns:
        timestamps = [kp['timestamp'] for kp in typing_patterns if 'timestamp' in kp and kp['timestamp'] is not None]
        dwell_times = [kp['dwellTime'] for kp in typing_patterns if 'dwellTime' in kp and kp['dwellTime'] is not None]

        if len(timestamps) > 1:
            typing_patterns.sort(key=lambda x: x['timestamp'])
            typing_duration = (timestamps[-1] - timestamps[0]) / 1000 # in seconds
            features['typing_duration_sec'] = typing_duration
            if typing_duration > 0:
                features['char_per_sec'] = len(typing_patterns) / typing_duration
            else:
                features['char_per_sec'] = 0

            flight_times = []
            for i in range(len(timestamps) - 1):
                if 'dwellTime' in typing_patterns[i] and typing_patterns[i]['dwellTime'] is not None:
                     flight_time_ms = typing_patterns[i+1]['timestamp'] - (typing_patterns[i]['timestamp'] + typing_patterns[i]['dwellTime'])
                     if flight_time_ms >= 0:
                         flight_times.append(flight_time_ms)

            if dwell_times:
                features['avg_dwell_time_ms'] = sum(dwell_times) / len(dwell_times)
                features['std_dev_dwell_time_ms'] = pd.Series(dwell_times).std() if len(dwell_times) > 1 else 0
            else:
                features['avg_dwell_time_ms'] = 0
                features['std_dev_dwell_time_ms'] = 0

            if flight_times:
                features['avg_flight_time_ms'] = sum(flight_times) / len(flight_times)
                features['std_dev_flight_time_ms'] = pd.Series(flight_times).std() if len(flight_times) > 1 else 0
            else:
                features['avg_flight_time_ms'] = 0
                features['std_dev_flight_time_ms'] = 0

        else:
            features['typing_duration_sec'] = 0
            features['char_per_sec'] = 0
            features['avg_dwell_time_ms'] = dwell_times[0] if dwell_times else 0
            features['std_dev_dwell_time_ms'] = 0
            features['avg_flight_time_ms'] = 0
            features['std_dev_flight_time_ms'] = 0

        backspaces = sum(1 for kp in typing_patterns if kp['key'] == 'Backspace')
        features['backspace_ratio'] = backspaces / len(typing_patterns) if len(typing_patterns) > 0 else 0
    else:
        features['typing_duration_sec'] = 0
        features['char_per_sec'] = 0
        features['avg_dwell_time_ms'] = 0
        features['std_dev_dwell_time_ms'] = 0
        features['avg_flight_time_ms'] = 0
        features['std_dev_flight_time_ms'] = 0
        features['backspace_ratio'] = 0

    # --- 2. Mouse Movements Features ---
    mouse_movements = behavior_data.get('mouseMovements', [])
    if mouse_movements and len(mouse_movements) > 1:
        mouse_movements.sort(key=lambda x: x['timestamp'])
        total_dist = 0
        speeds = []
        for i in range(len(mouse_movements) - 1):
            x1, y1, t1 = mouse_movements[i]['x'], mouse_movements[i]['y'], mouse_movements[i]['timestamp']
            x2, y2, t2 = mouse_movements[i+1]['x'], mouse_movements[i+1]['y'], mouse_movements[i+1]['timestamp']
            
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_dist += distance
            
            time_diff_sec = (t2 - t1) / 1000
            if time_diff_sec > 0:
                speeds.append(distance / time_diff_sec)

        features['total_mouse_distance_px'] = total_dist
        if speeds:
            features['avg_mouse_speed_px_sec'] = sum(speeds) / len(speeds)
            features['max_mouse_speed_px_sec'] = max(speeds)
            features['std_dev_mouse_speed_px_sec'] = pd.Series(speeds).std() if len(speeds) > 1 else 0
        else:
            features['avg_mouse_speed_px_sec'] = 0
            features['max_mouse_speed_px_sec'] = 0
            features['std_dev_mouse_speed_px_sec'] = 0

        xs = [m['x'] for m in mouse_movements]
        ys = [m['y'] for m in mouse_movements]
        if xs and ys:
            features['mouse_movement_width'] = max(xs) - min(xs)
            features['mouse_movement_height'] = max(ys) - min(ys)
            features['mouse_movement_area_px'] = features['mouse_movement_width'] * features['mouse_movement_height']
        else:
            features['mouse_movement_width'] = 0
            features['mouse_movement_height'] = 0
            features['mouse_movement_area_px'] = 0

    else:
        features['total_mouse_distance_px'] = 0
        features['avg_mouse_speed_px_sec'] = 0
        features['max_mouse_speed_px_sec'] = 0
        features['std_dev_mouse_speed_px_sec'] = 0
        features['mouse_movement_width'] = 0
        features['mouse_movement_height'] = 0
        features['mouse_movement_area_px'] = 0

    # --- 3. Click Events Features ---
    click_events = behavior_data.get('clickEvents', [])
    if click_events:
        features['num_clicks'] = len(click_events)
        
        hover_times = [ce['hoverTime'] for ce in click_events if 'hoverTime' in ce and ce['hoverTime'] is not None]
        if hover_times:
            features['avg_hover_time_ms'] = sum(hover_times) / len(hover_times)
            features['std_dev_hover_time_ms'] = pd.Series(hover_times).std() if len(hover_times) > 1 else 0
        else:
            features['avg_hover_time_ms'] = 0
            features['std_dev_hover_time_ms'] = 0

        off_target_clicks = sum(1 for ce in click_events if ce['target'] in ['HTML', 'BODY'])
        features['off_target_click_ratio'] = off_target_clicks / features['num_clicks']

        click_timestamps = sorted([ce['timestamp'] for ce in click_events if 'timestamp' in ce and ce['timestamp'] is not None])
        if len(click_timestamps) > 1:
            session_duration_clicks = (click_timestamps[-1] - click_timestamps[0]) / 1000
            if session_duration_clicks > 0:
                features['clicks_per_sec'] = features['num_clicks'] / session_duration_clicks
            else:
                features['clicks_per_sec'] = 0
            
            time_between_clicks = [(click_timestamps[i+1] - click_timestamps[i]) for i in range(len(click_timestamps) - 1)]
            if time_between_clicks:
                features['avg_time_between_clicks_ms'] = sum(time_between_clicks) / len(time_between_clicks)
                features['std_dev_time_between_clicks_ms'] = pd.Series(time_between_clicks).std() if len(time_between_clicks) > 1 else 0
            else:
                features['avg_time_between_clicks_ms'] = 0
                features['std_dev_time_between_clicks_ms'] = 0
        else:
            features['clicks_per_sec'] = 0
            features['avg_time_between_clicks_ms'] = 0
            features['std_dev_time_between_clicks_ms'] = 0

    else:
        features['num_clicks'] = 0
        features['avg_hover_time_ms'] = 0
        features['std_dev_hover_time_ms'] = 0
        features['off_target_click_ratio'] = 0
        features['clicks_per_sec'] = 0
        features['avg_time_between_clicks_ms'] = 0
        features['std_dev_time_between_clicks_ms'] = 0

    # --- 4. Device Info Features ---
    device_info = behavior_data.get('deviceInfo', {})
    
    if 'screenResolution' in device_info and device_info['screenResolution']:
        try:
            width, height = map(int, device_info['screenResolution'].split('x'))
            features['screen_width'] = width
            features['screen_height'] = height
            features['screen_aspect_ratio'] = width / height if height > 0 else 0
        except ValueError:
            features['screen_width'] = 0
            features['screen_height'] = 0
            features['screen_aspect_ratio'] = 0
    else:
        features['screen_width'] = 0
        features['screen_height'] = 0
        features['screen_aspect_ratio'] = 0
    
    if 'viewportSize' in device_info and device_info['viewportSize']:
        try:
            vw, vh = map(int, device_info['viewportSize'].split('x'))
            features['viewport_width'] = vw
            features['viewport_height'] = vh
            features['viewport_aspect_ratio'] = vw / vh if vh > 0 else 0
        except ValueError:
            features['viewport_width'] = 0
            features['viewport_height'] = 0
            features['viewport_aspect_ratio'] = 0
    else:
        features['viewport_width'] = 0
        features['viewport_height'] = 0
        features['viewport_aspect_ratio'] = 0

    features['browser'] = device_info.get('browser', 'unknown')
    features['os'] = device_info.get('os', 'unknown')
    features['timezone'] = device_info.get('timezone', 'unknown')

    # --- 5. Fingerprint Feature ---
    features['fingerprint_hash'] = hash(behavior_data.get('fingerprint', ''))

    return features

# --- Load Model and Scaler on Startup ---
@app.on_event("startup")
async def load_ml_assets():
    global ml_model, scaler, numerical_cols_to_scale, temporary_behavior_storage

    # Load the trained Isolation Forest model
    try:
        ml_model = joblib.load(MODEL_PATH)
        print(f"ML Model '{MODEL_PATH}' loaded successfully.")
    except FileNotFoundError:
        raise RuntimeError(f"ML Model file not found at {MODEL_PATH}. Ensure it's in the same directory.")
    except Exception as e:
        raise RuntimeError(f"Error loading ML model: {e}")

    # Load the processed data to fit the scaler to the same columns and scale
    try:
        df_processed = pd.read_csv(PROCESSED_DATA_PATH)
        # Drop the 'label' column and any one-hot encoded columns from the base X
        X_base = df_processed.drop('label', axis=1)
        
        # Identify numerical columns for scaling, excluding one-hot encoded ones
        # This logic should match what was used during training
        numerical_cols_to_scale = [
            col for col in X_base.columns 
            if X_base[col].dtype in ['float64', 'int64'] and not (X_base[col].isin([0, 1]).all() and X_base[col].nunique() <= 2)
        ]
        
        # Initialize and fit the scaler to the training data's numerical columns
        scaler = StandardScaler()
        scaler.fit(X_base[numerical_cols_to_scale])
        print("StandardScaler fitted successfully using processed training data.")

    except FileNotFoundError:
        raise RuntimeError(f"Processed data file not found at {PROCESSED_DATA_PATH}. Needed to fit StandardScaler.")
    except Exception as e:
        raise RuntimeError(f"Error initializing StandardScaler: {e}")

    # Initialize storage
    temporary_behavior_storage = load_behavior_logs()
    print(f"Loaded {len(temporary_behavior_storage)} existing behavior logs from {BEHAVIOR_LOG_FILE}")


@app.get("/")
async def root():
    return {"message": "BEACON Backend is running!"}

@app.post("/api/behavior")
async def receive_behavior_data(log_entry: BehaviorLogEntry, request: Request): # <--- CHANGED: Expecting full BehaviorLogEntry
    """
    Receives behavioral data from the frontend, processes it, and performs anomaly detection.
    Stores the raw data in a local file.
    """
    if ml_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="ML model or scaler not loaded. Server is not ready.")

    try:
        # Extract features from the incoming raw data using log_entry.data
        features_dict = extract_features(log_entry.data.model_dump(by_alias=True)) # <--- CHANGED: Access log_entry.data

        # Convert features dictionary to a Pandas DataFrame row
        # Ensure column order matches the training data by creating a dummy DataFrame
        # and then populating its values.
        # This requires knowing all possible columns (including one-hot encoded ones).
        # A robust way is to load the original processed_behavior_features.csv once more
        # or store the full list of columns.
        
        # For simplicity, let's derive columns from the initial X_base used for scaler.fit()
        # This assumes the categorical features won't introduce new categories not seen in training.
        
        # Create a DataFrame from the extracted features
        features_df = pd.DataFrame([features_dict])

        # Handle categorical features using one-hot encoding, matching training columns
        # Get list of all columns from the X_base used during scaler fit.
        # This needs to be precisely the columns the model was trained on.
        # A robust way is to save the feature columns order during training.
        # For now, let's re-create a dummy df from processed data to get correct column order.
        dummy_df_for_cols = pd.read_csv(PROCESSED_DATA_PATH).drop('label', axis=1)
        
        # Apply one-hot encoding on the new data, ensuring it aligns with training columns
        features_df_encoded = pd.get_dummies(features_df, columns=['browser', 'os', 'timezone'], prefix=['browser', 'os', 'tz'])

        # Align columns with the training data (dummy_df_for_cols). Add missing columns as 0.
        missing_cols = set(dummy_df_for_cols.columns) - set(features_df_encoded.columns)
        for c in missing_cols:
            features_df_encoded[c] = 0
        
        # Ensure the order of columns is the same as the training data
        features_final = features_df_encoded[dummy_df_for_cols.columns]

        # Apply scaling to the numerical columns
        features_final[numerical_cols_to_scale] = scaler.transform(features_final[numerical_cols_to_scale])

        # Make prediction and get anomaly score
        anomaly_prediction = ml_model.predict(features_final)[0] # -1 for anomaly, 1 for normal
        anomaly_score = ml_model.decision_function(features_final)[0] # Lower is more anomalous

        # Map prediction to 0 (normal) or 1 (anomalous)
        is_anomalous = 1 if anomaly_prediction == -1 else 0
        
        # Get user email for logging (already part of log_entry)
        user_email = log_entry.user_email if log_entry.user_email else "anonymous@example.com"
        
        # Log the raw incoming behavior data
        save_behavior_log(log_entry) # log_entry is already the correct full structure
        temporary_behavior_storage.append(log_entry) # For in-memory lookup if needed

        print(f"Received behavior data for {user_email}. Prediction: {'Anomalous' if is_anomalous else 'Normal'} (Score: {anomaly_score:.4f}).")
        
        return JSONResponse(content={
            "message": "Behavior data processed successfully",
            "status": "success",
            "is_anomalous": bool(is_anomalous), # Convert to boolean for frontend
            "anomaly_score": float(anomaly_score)
        }, status_code=200)

    except Exception as e:
        print(f"Error processing behavior data: {e}")
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@app.get("/api/behavior_logs")
async def get_behavior_logs():
    """
    Retrieves all temporarily stored raw behavior logs from memory.
    (For demonstration/debugging purposes)
    """
    return temporary_behavior_storage

# To run the application directly from this script
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)