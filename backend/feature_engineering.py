import pandas as pd
import json
import math

# --- Feature Extraction Function ---
def extract_features(behavior_data):
    features = {}

    # --- 1. Typing Patterns Features ---
    typing_patterns = behavior_data.get('typingPatterns', [])
    if typing_patterns:
        timestamps = [kp['timestamp'] for kp in typing_patterns if 'timestamp' in kp and kp['timestamp'] is not None]
        dwell_times = [kp['dwellTime'] for kp in typing_patterns if 'dwellTime' in kp and kp['dwellTime'] is not None]

        if len(timestamps) > 1:
            # Sort by timestamp to ensure correct order for calculations
            typing_patterns.sort(key=lambda x: x['timestamp'])

            # Calculate total typing duration and speed
            typing_duration = (timestamps[-1] - timestamps[0]) / 1000 # in seconds
            features['typing_duration_sec'] = typing_duration
            if typing_duration > 0:
                features['char_per_sec'] = len(typing_patterns) / typing_duration
            else:
                features['char_per_sec'] = 0

            # Flight times
            flight_times = []
            for i in range(len(timestamps) - 1):
                # Flight time is the duration between key up of current and key down of next
                # Assuming dwellTime is already available from tracker, otherwise needs keyUp timestamp
                if 'dwellTime' in typing_patterns[i] and typing_patterns[i]['dwellTime'] is not None:
                     flight_time_ms = typing_patterns[i+1]['timestamp'] - (typing_patterns[i]['timestamp'] + typing_patterns[i]['dwellTime'])
                     if flight_time_ms >= 0: # Ensure non-negative flight time
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

        else: # Handle case with 0 or 1 keypress
            features['typing_duration_sec'] = 0
            features['char_per_sec'] = 0
            features['avg_dwell_time_ms'] = dwell_times[0] if dwell_times else 0
            features['std_dev_dwell_time_ms'] = 0
            features['avg_flight_time_ms'] = 0
            features['std_dev_flight_time_ms'] = 0

        # Error rate
        backspaces = sum(1 for kp in typing_patterns if kp['key'] == 'Backspace')
        features['backspace_ratio'] = backspaces / len(typing_patterns) if len(typing_patterns) > 0 else 0
    else: # No typing data
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

        # Mouse movement area (simple bounding box)
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

    else: # No mouse movement data
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

        # Click accuracy (example: clicks on 'HTML' or 'BODY' often indicate off-target)
        off_target_clicks = sum(1 for ce in click_events if ce['target'] in ['HTML', 'BODY'])
        features['off_target_click_ratio'] = off_target_clicks / features['num_clicks']

        # Click rates / time between clicks
        click_timestamps = sorted([ce['timestamp'] for ce in click_events if 'timestamp' in ce and ce['timestamp'] is not None])
        if len(click_timestamps) > 1:
            session_duration_clicks = (click_timestamps[-1] - click_timestamps[0]) / 1000 # in seconds
            if session_duration_clicks > 0:
                features['clicks_per_sec'] = features['num_clicks'] / session_duration_clicks
            else:
                features['clicks_per_sec'] = 0 # All clicks at same timestamp
            
            time_between_clicks = [(click_timestamps[i+1] - click_timestamps[i]) for i in range(len(click_timestamps) - 1)]
            if time_between_clicks:
                features['avg_time_between_clicks_ms'] = sum(time_between_clicks) / len(time_between_clicks)
                features['std_dev_time_between_clicks_ms'] = pd.Series(time_between_clicks).std() if len(time_between_clicks) > 1 else 0
            else:
                features['avg_time_between_clicks_ms'] = 0
                features['std_dev_time_between_clicks_ms'] = 0
        else: # 0 or 1 click
            features['clicks_per_sec'] = 0
            features['avg_time_between_clicks_ms'] = 0
            features['std_dev_time_between_clicks_ms'] = 0

    else: # No click data
        features['num_clicks'] = 0
        features['avg_hover_time_ms'] = 0
        features['std_dev_hover_time_ms'] = 0
        features['off_target_click_ratio'] = 0
        features['clicks_per_sec'] = 0
        features['avg_time_between_clicks_ms'] = 0
        features['std_dev_time_between_clicks_ms'] = 0

    # --- 4. Device Info Features ---
    device_info = behavior_data.get('deviceInfo', {})
    
    # Screen Resolution
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
    
    # Viewport Size
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

    # Categorical features for one-hot encoding later (or direct use if model supports)
    features['browser'] = device_info.get('browser', 'unknown')
    features['os'] = device_info.get('os', 'unknown')
    features['timezone'] = device_info.get('timezone', 'unknown')

    # --- 5. Fingerprint Feature ---
    # For now, we'll simply use a numerical hash of the fingerprint string.
    # In a real system, you'd compare against known fingerprints for a user.
    features['fingerprint_hash'] = hash(behavior_data.get('fingerprint', '')) # Python's hash can vary, use a more stable one if needed for persistence

    return features

# --- Main Script Execution ---
if __name__ == "__main__":
    # Load your combined dataset
    # Make sure 'behavior_combined_large.json' is in the same directory as this script,
    # or provide the full path to it.
    try:
        with open('behavior_combined_large.json', 'r') as f:
            raw_data = json.load(f)
        print(f"Loaded {len(raw_data)} raw behavior samples.")
    except FileNotFoundError:
        print("Error: 'behavior_combined_large.json' not found.")
        print("Please ensure the file is in the same directory or provide the correct path.")
        exit()

    features_list = []
    labels = []

    print("Extracting features from each sample...")
    for entry in raw_data:
        try:
            features = extract_features(entry['data'])
            features_list.append(features)
            labels.append(entry['label'])
        except Exception as e:
            print(f"Error extracting features for an entry: {e}")
            # Optionally, log the problematic entry or skip it
            continue

    # Create a Pandas DataFrame
    df = pd.DataFrame(features_list)
    df['label'] = labels

    # --- Data Cleaning and Preparation ---
    # Handle infinite values (e.g., from division by zero)
    df.replace([float('inf'), float('-inf')], 0, inplace=True) # Replace inf with 0, or NaN for later imputation

    # Fill any remaining NaN values (e.g., if a list was empty and std() was called)
    # A simple strategy is to fill with 0, but consider median or mean for numerical features.
    # For this example, we'll fill with 0.
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)


    # One-hot encode categorical features (browser, os, timezone)
    # This creates new columns for each unique category
    df = pd.get_dummies(df, columns=['browser', 'os', 'timezone'], prefix=['browser', 'os', 'tz'])

    # Convert boolean columns created by get_dummies to int (0 or 1) if desired
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

    # Convert labels to numerical: 1 for 'anomalous', 0 for 'normal'
    df['label'] = df['label'].apply(lambda x: 1 if x == 'anomalous' else 0)

    print("\n--- Feature Engineering Complete ---")
    print("\nDataFrame Head:")
    print(df.head())

    print("\nDataFrame Info:")
    df.info()

    print("\nLabel Distribution:")
    print(df['label'].value_counts())

    # Save the processed DataFrame to a CSV file for later use
    df.to_csv("processed_behavior_features.csv", index=False)
    print("\nProcessed features saved to 'processed_behavior_features.csv'")

    print("\nYour DataFrame 'df' is now ready for model training!")
    print("X (features) will be df.drop('label', axis=1)")
    print("y (labels) will be df['label']")