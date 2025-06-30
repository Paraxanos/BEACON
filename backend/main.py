# BEACON Backend with full feature extraction and batch CSV generation
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import datetime
import json
import os
import joblib
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from fastapi.middleware.cors import CORSMiddleware

# --- Configuration ---
BEHAVIOR_LOG_FILE = "behavior_logs.jsonl"
MODEL_PATH = "isolation_forest_model.joblib"
SCALER_PATH = "standard_scaler.joblib"
PROCESSED_DATA_PATH = "processed_behavior_features.csv"

app = FastAPI(
    title="BEACON Behavioral Backend",
    description="API for receiving and processing behavioral data from the BEACON web app.",
    version="0.1.0"
)

origins = ["http://localhost:5173", "http://127.0.0.1:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BehaviorDataPayload(BaseModel):
    typingPatterns: Optional[List[Dict[str, Any]]] = None
    mouseMovements: Optional[List[Dict[str, Any]]] = None
    clickEvents: Optional[List[Dict[str, Any]]] = None
    deviceInfo: Dict[str, Any]
    fingerprint: Optional[str] = None

class BehaviorLogEntry(BaseModel):
    timestamp: str
    user_email: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    data: BehaviorDataPayload

ml_model = None
scaler = None
numerical_cols_to_scale: List[str] = []
all_feature_columns: List[str] = []
temporary_behavior_storage: List[BehaviorLogEntry] = []

def load_behavior_logs() -> List[BehaviorLogEntry]:
    logs = []
    if os.path.exists(BEHAVIOR_LOG_FILE):
        with open(BEHAVIOR_LOG_FILE, 'r') as f:
            for line in f:
                try:
                    log_data = json.loads(line.strip())
                    logs.append(BehaviorLogEntry(**log_data))
                except Exception as e:
                    print(f"Log load error: {e}")
    return logs

def save_behavior_log(log_entry: BehaviorLogEntry):
    with open(BEHAVIOR_LOG_FILE, 'a') as f:
        f.write(json.dumps(log_entry.model_dump(by_alias=True)) + '\n')

def extract_features(behavior_data: Dict[str, Any], context_data: Optional[Dict[str, Any]] = None, timestamp_str: Optional[str] = None) -> Dict[str, Any]:
    features = {}
    
    # Typing pattern features
    typing_patterns = behavior_data.get('typingPatterns', [])
    if typing_patterns:
        typing_patterns.sort(key=lambda x: x['timestamp'])
        timestamps = [kp['timestamp'] for kp in typing_patterns if kp.get('timestamp') is not None]
        dwell_times = [kp['dwellTime'] for kp in typing_patterns if kp.get('dwellTime') is not None]

        if len(timestamps) > 1:
            typing_duration = (timestamps[-1] - timestamps[0]) / 1000
            features['typing_duration_sec'] = typing_duration
            features['char_per_sec'] = len(typing_patterns) / typing_duration if typing_duration > 0 else 0
            flight_times = [typing_patterns[i+1]['timestamp'] - (typing_patterns[i]['timestamp'] + typing_patterns[i]['dwellTime'])
                            for i in range(len(timestamps) - 1)
                            if 'dwellTime' in typing_patterns[i] and typing_patterns[i]['dwellTime'] is not None]
            features['avg_dwell_time_ms'] = np.mean(dwell_times)
            features['std_dev_dwell_time_ms'] = np.std(dwell_times)
            features['avg_flight_time_ms'] = np.mean(flight_times) if flight_times else 0
            features['std_dev_flight_time_ms'] = np.std(flight_times) if flight_times else 0
        else:
            features.update({
                'typing_duration_sec': 0,
                'char_per_sec': 0,
                'avg_dwell_time_ms': dwell_times[0] if dwell_times else 0,
                'std_dev_dwell_time_ms': 0,
                'avg_flight_time_ms': 0,
                'std_dev_flight_time_ms': 0
            })

        backspaces = sum(1 for kp in typing_patterns if kp.get('key') == 'Backspace')
        features['backspace_ratio'] = backspaces / len(typing_patterns)
    else:
        features.update({k: 0 for k in [
            'typing_duration_sec', 'char_per_sec', 'avg_dwell_time_ms', 'std_dev_dwell_time_ms',
            'avg_flight_time_ms', 'std_dev_flight_time_ms', 'backspace_ratio']})

    # Mouse movement features
    mouse_movements = behavior_data.get('mouseMovements', [])
    if mouse_movements and len(mouse_movements) > 1:
        mouse_movements.sort(key=lambda x: x['timestamp'])
        total_dist = 0
        speeds = []
        for i in range(len(mouse_movements) - 1):
            x1, y1, t1 = mouse_movements[i]['x'], mouse_movements[i]['y'], mouse_movements[i]['timestamp']
            x2, y2, t2 = mouse_movements[i+1]['x'], mouse_movements[i+1]['y'], mouse_movements[i+1]['timestamp']
            dist = math.hypot(x2 - x1, y2 - y1)
            total_dist += dist
            time_diff = (t2 - t1) / 1000
            if time_diff > 0:
                speeds.append(dist / time_diff)

        xs = [m['x'] for m in mouse_movements]
        ys = [m['y'] for m in mouse_movements]
        features.update({
            'total_mouse_distance_px': total_dist,
            'avg_mouse_speed_px_sec': np.mean(speeds) if speeds else 0,
            'max_mouse_speed_px_sec': max(speeds) if speeds else 0,
            'std_dev_mouse_speed_px_sec': np.std(speeds) if len(speeds) > 1 else 0,
            'mouse_movement_width': max(xs) - min(xs),
            'mouse_movement_height': max(ys) - min(ys),
            'mouse_movement_area_px': (max(xs) - min(xs)) * (max(ys) - min(ys))
        })
    else:
        features.update({k: 0 for k in [
            'total_mouse_distance_px', 'avg_mouse_speed_px_sec', 'max_mouse_speed_px_sec',
            'std_dev_mouse_speed_px_sec', 'mouse_movement_width', 'mouse_movement_height', 'mouse_movement_area_px']})

    # Click features
    click_events = behavior_data.get('clickEvents', [])
    if click_events:
        hovers = [e['hoverTime'] for e in click_events if 'hoverTime' in e and e['hoverTime'] is not None]
        click_times = sorted(e['timestamp'] for e in click_events if 'timestamp' in e)
        inter_click = [click_times[i+1] - click_times[i] for i in range(len(click_times) - 1)]
        features.update({
            'num_clicks': len(click_events),
            'avg_hover_time_ms': np.mean(hovers) if hovers else 0,
            'std_dev_hover_time_ms': np.std(hovers) if len(hovers) > 1 else 0,
            'off_target_click_ratio': sum(1 for e in click_events if e['target'] in ['HTML', 'BODY']) / len(click_events),
            'clicks_per_sec': len(click_events) / ((click_times[-1] - click_times[0]) / 1000) if len(click_times) > 1 else 0,
            'avg_time_between_clicks_ms': np.mean(inter_click) if inter_click else 0,
            'std_dev_time_between_clicks_ms': np.std(inter_click) if len(inter_click) > 1 else 0
        })
    else:
        features.update({k: 0 for k in [
            'num_clicks', 'avg_hover_time_ms', 'std_dev_hover_time_ms', 'off_target_click_ratio',
            'clicks_per_sec', 'avg_time_between_clicks_ms', 'std_dev_time_between_clicks_ms']})

    # Device info
    device_info = behavior_data.get('deviceInfo', {})
    if 'screenResolution' in device_info:
        try:
            w, h = map(int, device_info['screenResolution'].split('x'))
            features.update({
                'screen_width': w,
                'screen_height': h,
                'screen_aspect_ratio': w / h if h else 0
            })
        except:
            features.update({'screen_width': 0, 'screen_height': 0, 'screen_aspect_ratio': 0})
    else:
        features.update({'screen_width': 0, 'screen_height': 0, 'screen_aspect_ratio': 0})

    if 'viewportSize' in device_info:
        try:
            vw, vh = map(int, device_info['viewportSize'].split('x'))
            features.update({
                'viewport_width': vw,
                'viewport_height': vh,
                'viewport_aspect_ratio': vw / vh if vh else 0
            })
        except:
            features.update({'viewport_width': 0, 'viewport_height': 0, 'viewport_aspect_ratio': 0})
    else:
        features.update({'viewport_width': 0, 'viewport_height': 0, 'viewport_aspect_ratio': 0})

    features['browser'] = device_info.get('browser', 'unknown').lower()
    features['os'] = device_info.get('os', 'unknown').lower()
    features['timezone'] = context_data.get('timezone') if context_data else device_info.get('timezone', 'unknown')

    features['fingerprint_hash'] = hash(behavior_data.get('fingerprint', ''))

    # Temporal features
    if timestamp_str:
        try:
            dt = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            features['time_of_day_sin'] = np.sin(2 * np.pi * dt.hour / 24)
            features['time_of_day_cos'] = np.cos(2 * np.pi * dt.hour / 24)
            features['day_of_week_sin'] = np.sin(2 * np.pi * dt.weekday() / 7)
            features['day_of_week_cos'] = np.cos(2 * np.pi * dt.weekday() / 7)
        except:
            features.update({k: 0 for k in ['time_of_day_sin', 'time_of_day_cos', 'day_of_week_sin', 'day_of_week_cos']})
    else:
        features.update({k: 0 for k in ['time_of_day_sin', 'time_of_day_cos', 'day_of_week_sin', 'day_of_week_cos']})

    # Geolocation features
    if context_data and 'location' in context_data:
        loc = context_data['location']
        if 'latitude' in loc and 'longitude' in loc:
            def haversine(lat1, lon1, lat2, lon2):
                R = 6371
                lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                return R * c

            trusted_lat, trusted_lon = 17.4375, 78.4482
            dist = haversine(trusted_lat, trusted_lon, loc['latitude'], loc['longitude'])
            features['distance_from_trusted_loc'] = dist
            features['is_impossible_travel'] = int(dist > 100)
        else:
            features.update({'distance_from_trusted_loc': 0, 'is_impossible_travel': 0})
    else:
        features.update({'distance_from_trusted_loc': 0, 'is_impossible_travel': 0})

    return features



# --- Load Model and Scaler on Startup ---
@app.on_event("startup")
async def load_ml_assets():
    global ml_model, scaler, numerical_cols_to_scale, all_feature_columns, temporary_behavior_storage

    # Load the trained Isolation Forest model
    try:
        ml_model = joblib.load(MODEL_PATH)
        print(f"ML Model '{MODEL_PATH}' loaded successfully.")
    except FileNotFoundError:
        raise RuntimeError(f"ML Model file not found at {MODEL_PATH}. Ensure it's in the same directory.")
    except Exception as e:
        raise RuntimeError(f"Error loading ML model: {e}")

    # Load the processed data to fit the scaler and get the exact column order
    try:
        df_processed = pd.read_csv(PROCESSED_DATA_PATH)
        X_base = df_processed.drop('label', axis=1)
        
        # Store the exact order of all columns the model was trained on
        all_feature_columns = X_base.columns.tolist()

        # Identify numerical columns for scaling, excluding one-hot encoded ones
        # These are the numerical columns from the provided CSV
        numerical_cols_to_scale = [
            'typing_duration_sec', 'char_per_sec', 'avg_dwell_time_ms', 
            'std_dev_dwell_time_ms', 'avg_flight_time_ms', 'std_dev_flight_time_ms', 
            'backspace_ratio', 'total_mouse_distance_px', 'avg_mouse_speed_px_sec', 
            'max_mouse_speed_px_sec', 'std_dev_mouse_speed_px_sec', 'mouse_movement_width', 
            'mouse_movement_height', 'mouse_movement_area_px', 'num_clicks', 
            'avg_hover_time_ms', 'std_dev_hover_time_ms', 'off_target_click_ratio', 
            'clicks_per_sec', 'avg_time_between_clicks_ms', 'std_dev_time_between_clicks_ms', 
            'screen_width', 'screen_height', 'screen_aspect_ratio', 'viewport_width', 
            'viewport_height', 'viewport_aspect_ratio', 'fingerprint_hash'
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
async def receive_behavior_data(log_entry: BehaviorLogEntry, request: Request):
    """
    Receives behavioral data from the frontend, processes it, and performs anomaly detection.
    Stores the raw data in a local file.
    """
    if ml_model is None or scaler is None or not all_feature_columns:
        raise HTTPException(status_code=503, detail="ML model, scaler, or feature columns not loaded. Server is not ready.")

    try:
        # Extract features from the incoming raw data and context
        # Note: Geolocation and temporal features are NOT extracted here
        # because they were not present in the provided processed_behavior_features.csv
        features_dict = extract_features(
            log_entry.data.model_dump(by_alias=True), 
            log_entry.context # Pass context, though geo/temporal parts are ignored in extract_features now
        )

        # Convert features dictionary to a Pandas DataFrame row
        features_df = pd.DataFrame([features_dict])

        # Handle categorical features using one-hot encoding, ensuring alignment with training columns
        # Get list of all columns from the X_base used during scaler fit.
        # This needs to be precisely the columns the model was trained on.
        features_df_encoded = pd.get_dummies(features_df, columns=['browser', 'os', 'timezone'], prefix=['browser', 'os', 'tz'])

        # Align columns with the training data (all_feature_columns). Add missing columns as 0.
        missing_cols = set(all_feature_columns) - set(features_df_encoded.columns)
        for c in missing_cols:
            features_df_encoded[c] = 0
        
        # Ensure the order of columns is the same as the training data
        features_final = features_df_encoded[all_feature_columns]

        # Apply scaling to the numerical columns
        features_final[numerical_cols_to_scale] = scaler.transform(features_final[numerical_cols_to_scale])

        # Make prediction and get anomaly score
        anomaly_prediction = ml_model.predict(features_final)[0] # -1 for anomaly, 1 for normal
        anomaly_score = ml_model.decision_function(features_final)[0] # Lower is more anomalous

        # Map prediction to 0 (normal) or 1 (anomalous)
        is_anomalous_from_model = 1 if anomaly_prediction == -1 else 0
        
        # --- Apply Shadow Mode logic to the response ---
        # Access safeMode from the context dictionary
        is_safe_mode = log_entry.context.get('safeMode', False) if log_entry.context else False
        
        if is_safe_mode:
            # In safe mode, we log the prediction but always report a 'normal' status to the frontend
            print(f"Shadow Mode active. Prediction was {'Anomalous' if is_anomalous_from_model else 'Normal'} (Score: {anomaly_score:.4f}) but reporting 'Normal'.")
            is_anomalous_for_frontend = False
            # Report a neutral score (e.g., 0.5) when in safe mode
            anomaly_score_for_frontend = 0.5 
        else:
            # Not in safe mode, report the actual model prediction
            is_anomalous_for_frontend = bool(is_anomalous_from_model)
            # Convert the Isolation Forest score to a 0-1 scale for the frontend UI
            # Scores are typically between -1 and 1, with 0 being the threshold.
            # We can map them to 0-1 for a trust score.
            # A simple linear mapping: score = (raw_score - min_score) / (max_score - min_score)
            # For Isolation Forest, a common heuristic is to map from [-0.1, 0.1] to [0, 1]
            # where -0.1 is very anomalous and 0.1 is very normal.
            # Let's use a simple mapping that makes negative scores lower trust and positive scores higher.
            # A score of 0 (threshold) maps to 0.5 trust.
            anomaly_score_for_frontend = (anomaly_score + 1) / 2 # Maps [-1, 1] to [0, 1]
            anomaly_score_for_frontend = max(0, min(1, anomaly_score_for_frontend)) # Clamp to 0-1 range
            
        # Get user email for logging (already part of log_entry)
        user_email = log_entry.user_email if log_entry.user_email else "anonymous@example.com"
        
        # Log the raw incoming behavior data
        save_behavior_log(log_entry) # log_entry is already the correct full structure
        temporary_behavior_storage.append(log_entry) # For in-memory lookup if needed

        print(f"Received behavior data for {user_email}. Model Prediction: {'Anomalous' if is_anomalous_from_model else 'Normal'} (Score: {anomaly_score:.4f}). Reported to frontend: {'Anomalous' if is_anomalous_for_frontend else 'Normal'} (Score: {anomaly_score_for_frontend:.4f}).")
        
        return JSONResponse(content={
            "message": "Behavior data processed successfully",
            "status": "success",
            "is_anomalous": is_anomalous_for_frontend, # Use the modified value for the frontend
            "anomaly_score": float(anomaly_score_for_frontend) # Use the modified value for the frontend
        }, status_code=200)

    except Exception as e:
        print(f"Error processing behavior data: {e}")
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# --- Behavioral Drift (Retraining) Logic ---
def retrain_model_on_logs(user_email: Optional[str] = None):
    """
    Function to retrain the Isolation Forest model using data from the logs.
    Can be filtered by user_email for personalized models.
    This is a long-running task, so it should be run in the background.
    """
    print(f"Starting model retraining process for user: {user_email if user_email else 'All Users'}...")
    try:
        # 1. Load all logs from the JSONL file
        all_logs = load_behavior_logs()
        
        # Filter logs if a specific user_email is provided
        if user_email:
            logs_for_training = [log for log in all_logs if log.user_email == user_email]
            if not logs_for_training:
                print(f"No logs found for user {user_email} to retrain the model.")
                return
        else:
            logs_for_training = all_logs

        if not logs_for_training:
            print("No new logs to retrain the model with.")
            return

        # 2. Extract features from all logs
        # This will be memory intensive for very large datasets; consider batch processing or DB queries
        features_list = []
        labels_list = [] # We'll need labels if we want to retrain with contamination
        for log_entry in logs_for_training:
            # Pass the behavioral data and context data to extract_features
            # Note: Geolocation and temporal features are NOT extracted here
            features = extract_features(log_entry.data.model_dump(by_alias=True), log_entry.context)
            features_list.append(features)
            # For retraining, we assume all historical data is 'normal' for unsupervised IF,
            # or we need a labeling mechanism for supervised retraining.
            # For simplicity here, we'll treat all as normal for re-fitting the scaler.
            # If you have a way to label historical data, you'd use that here.
            labels_list.append(0) # Assume historical data is mostly 'normal' for unsupervised retraining

        full_df = pd.DataFrame(features_list)
        full_df['label'] = labels_list # Add a dummy label column for consistency with training script

        # Handle categorical features using one-hot encoding, matching all_feature_columns
        full_df_encoded = pd.get_dummies(full_df, columns=['browser', 'os', 'timezone'], prefix=['browser', 'os', 'tz'])

        # Align columns with the global all_feature_columns
        missing_cols_df = set(all_feature_columns) - set(full_df_encoded.columns)
        for c in missing_cols_df:
            full_df_encoded[c] = 0
        
        # Ensure the order of columns is the same as the original training data
        X_retrain = full_df_encoded[all_feature_columns]

        # 3. Re-fit the StandardScaler on the new combined data
        # Create a new scaler for retraining to ensure it's fresh
        new_scaler = StandardScaler()
        X_retrain_scaled = X_retrain.copy()
        X_retrain_scaled[numerical_cols_to_scale] = new_scaler.fit_transform(X_retrain[numerical_cols_to_scale])
        
        # 4. Re-train the Isolation Forest model
        # Recalculate contamination based on the new dataset if you have labeled data,
        # otherwise use a fixed value or the original one.
        # For unsupervised retraining, contamination can be a fixed small value (e.g., 0.01)
        # or derived from the proportion of 'anomalous' flags in the *logged* data if available.
        
        # Let's use a fixed contamination for unsupervised retraining, or calculate if labels exist
        # If you have a labeling process for historical logs, use:
        # contamination_rate = pd.Series(labels_list).value_counts(normalize=True).get(1, 0.01)
        # Otherwise, assume a small percentage of anomalies for unsupervised learning.
        contamination_rate_for_retrain = 0.01 # Assume 1% outliers in the 'normal' historical data

        new_model = IsolationForest(n_estimators=100, contamination=contamination_rate_for_retrain, random_state=42)
        new_model.fit(X_retrain_scaled)
        
        # 5. Save the new model and scaler
        joblib.dump(new_model, MODEL_PATH)
        joblib.dump(new_scaler, SCALER_PATH) # Save the new scaler
        
        # Update global variables for the running application
        global ml_model, scaler
        ml_model = new_model
        scaler = new_scaler

        print("Model retraining complete! New model and scaler saved and loaded into memory.")
        
    except Exception as e:
        print(f"Error during model retraining: {e}")
        import traceback
    
# Endpoint to trigger retraining
@app.post("/api/retrain_model")
async def trigger_retrain(background_tasks: BackgroundTasks, user_email: Optional[str] = None):
    """
    Triggers a background task to retrain the model using logged data.
    Can retrain for a specific user or all users.
    """
    background_tasks.add_task(retrain_model_on_logs, user_email=user_email)
    return {"message": f"Model retraining has been triggered in the background for user {user_email if user_email else 'all users'}. Check server logs for status."}

@app.get("/api/behavior_logs")
async def get_behavior_logs():
    """
    Retrieves all temporarily stored raw behavior logs from memory.
    (For demonstration/debugging purposes)
    """
    return temporary_behavior_storage

@app.get("/api/user_anomalies/{user_email}")
async def get_user_anomalies(user_email: str):
    """
    Returns all logs and anomaly results for a specific user.
    Useful for personalized anomaly tracking in dashboards.
    """
    user_logs = [log for log in temporary_behavior_storage if log.user_email == user_email]
    response = []

    for log in user_logs:
        features = extract_features(log.data.model_dump(), log.context, log.timestamp)
        df = pd.DataFrame([features])
        df = pd.get_dummies(df)
        for col in all_feature_columns:
            if col not in df:
                df[col] = 0
        df = df[all_feature_columns]
        df[numerical_cols_to_scale] = scaler.transform(df[numerical_cols_to_scale])
        pred = ml_model.predict(df)[0]
        score = ml_model.decision_function(df)[0]
        response.append({
            "timestamp": log.timestamp,
            "anomaly": pred == -1,
            "anomaly_score": round((score + 1) / 2, 4),
            "context": log.context
        })

    return {"user": user_email, "logs": response}

@app.get("/api/dashboard_summary")
async def get_dashboard_summary():
    """
    Returns a summary of recent anomaly trends for visualization.
    """
    from collections import defaultdict
    count_by_user = defaultdict(lambda: {"total": 0, "anomalies": 0})

    for log in temporary_behavior_storage:
        email = log.user_email or "unknown"
        count_by_user[email]["total"] += 1
        features = extract_features(log.data.model_dump(), log.context, log.timestamp)
        df = pd.DataFrame([features])
        df = pd.get_dummies(df)
        for col in all_feature_columns:
            if col not in df:
                df[col] = 0
        df = df[all_feature_columns]
        df[numerical_cols_to_scale] = scaler.transform(df[numerical_cols_to_scale])
        pred = ml_model.predict(df)[0]
        if pred == -1:
            count_by_user[email]["anomalies"] += 1

    return count_by_user

# To run the application directly from this script
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)