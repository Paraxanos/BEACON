import json
import random
import time
from datetime import datetime, timezone
import uuid # For generating unique fingerprints

# --- Configuration ---
NUM_NORMAL_SAMPLES = 1000
NUM_ANOMALOUS_SAMPLES = 200 # Typically fewer anomalous samples

# --- Helper functions to generate random data points ---

def generate_typing_pattern(is_anomalous=False):
    pattern = []
    keys = "abcdefghijklmnopqrstuvwxyz1234567890 "
    start_ts = int(time.time() * 1000)

    num_keys = random.randint(10, 50) if not is_anomalous else random.randint(5, 100) # Shorter/longer anomalous sequences
    
    for i in range(num_keys):
        key = random.choice(keys)
        code = f"Key{key.upper()}" if key.isalpha() else "Digit" + key if key.isdigit() else "Space"
        timestamp = start_ts + i * random.randint(50, 150) # Normal flight time

        dwell_time = random.randint(50, 100) # Normal dwell time
        if is_anomalous:
            # Simulate very fast (bot) or very slow/hesitant typing
            if random.random() < 0.5: # 50% chance for fast/slow anomaly
                dwell_time = random.randint(10, 40) # Very fast
                timestamp = start_ts + i * random.randint(5, 30) # Very short flight time
            else:
                dwell_time = random.randint(200, 500) # Very slow
                timestamp = start_ts + i * random.randint(200, 800) # Very long flight time

            # Add more backspaces for anomalous
            if random.random() < 0.15 and i > 0: # 15% chance to add a backspace
                pattern.append({
                    "key": "Backspace", "code": "Backspace", "timestamp": timestamp - random.randint(10,50), "dwellTime": random.randint(50,100)
                })

        pattern.append({
            "key": key,
            "code": code,
            "timestamp": timestamp,
            "dwellTime": dwell_time
        })
    return pattern

def generate_mouse_movements(is_anomalous=False):
    movements = []
    start_ts = int(time.time() * 1000)
    
    num_moves = random.randint(20, 100) if not is_anomalous else random.randint(5, 200) # Vary movement count

    current_x, current_y = random.randint(0, 1920), random.randint(0, 1080) # Initial position
    
    for i in range(num_moves):
        timestamp = start_ts + i * random.randint(10, 50) # Normal speed
        
        if not is_anomalous:
            # Smooth, gradual movement
            target_x = current_x + random.randint(-20, 20)
            target_y = current_y + random.randint(-20, 20)
        else:
            if random.random() < 0.3: # Simulate jumps
                target_x = random.randint(0, 1920)
                target_y = random.randint(0, 1080)
            elif random.random() < 0.6: # Simulate erratic small movements
                target_x = current_x + random.randint(-5, 5)
                target_y = current_y + random.randint(-5, 5)
                timestamp = start_ts + i * random.randint(1, 10) # Very fast small movements
            else: # Simulate straight lines (bot-like)
                target_x = current_x + (random.choice([-1, 1]) * random.randint(10, 50))
                target_y = current_y # Only move horizontally for a bit

        movements.append({"x": max(0, min(1920, target_x)), "y": max(0, min(1080, target_y)), "timestamp": timestamp})
        current_x, current_y = target_x, target_y
        
    return movements

def generate_click_events(is_anomalous=False):
    clicks = []
    targets = ["BUTTON", "INPUT", "A", "DIV", "SPAN"]
    start_ts = int(time.time() * 1000)

    num_clicks = random.randint(2, 10) if not is_anomalous else random.randint(1, 20) # More clicks for anomalous
    
    for i in range(num_clicks):
        timestamp = start_ts + i * random.randint(500, 2000) # Normal interval
        x, y = random.randint(50, 1000), random.randint(50, 600)
        target = random.choice(targets)
        hover_time = random.randint(100, 500) # Normal hover
        
        if is_anomalous:
            if random.random() < 0.4: # Rapid clicks
                timestamp = start_ts + i * random.randint(10, 100)
                hover_time = random.randint(0, 50) # Very short/no hover
            elif random.random() < 0.8: # Off-target clicks
                x, y = random.randint(1, 40), random.randint(1, 40) # Edge of screen
                target = "HTML" # Clicking on nothing specific
        
        clicks.append({
            "x": x, "y": y, "target": target, "timestamp": timestamp, "hoverTime": hover_time
        })
    return clicks

def generate_device_info(is_anomalous=False):
    browsers = ["Chrome", "Firefox", "Edge", "Safari"]
    oss = ["Windows", "macOS", "Linux"]
    resolutions = ["1920x1080", "1440x900", "1366x768"]
    timezones = ["Asia/Kolkata", "America/New_York", "Europe/London"]

    info = {
        "browser": random.choice(browsers),
        "screenResolution": random.choice(resolutions),
        "viewportSize": f"{random.randint(800, 1600)}x{random.randint(600, 900)}",
        "os": random.choice(oss),
        "timezone": random.choice(timezones)
    }

    if is_anomalous and random.random() < 0.7: # High chance of device info mismatch
        # Change multiple attributes to simulate a different device/location
        info["browser"] = random.choice([b for b in browsers if b != info["browser"]])
        info["os"] = random.choice([o for o in oss if o != info["os"]])
        info["timezone"] = random.choice([t for t in timezones if t != info["timezone"]])
        info["screenResolution"] = random.choice([r for r in resolutions if r != info["screenResolution"]])

    return info

def generate_fingerprint():
    # Simple mock fingerprint ID
    return str(uuid.uuid4())

# --- Main generation function ---

def generate_behavior_sample(label_mode="normal"):
    timestamp_iso = datetime.now(timezone.utc).isoformat()
    is_anomalous = (label_mode == "anomalous")
    
    # Define a "trusted" location (e.g., Secunderabad)
    trusted_latitude = 17.4375
    trusted_longitude = 78.4482

    # Generate geolocation data
    geolocation = {
        "latitude": trusted_latitude + random.uniform(-0.1, 0.1), # Small variation around trusted loc
        "longitude": trusted_longitude + random.uniform(-0.1, 0.1)
    }
    if is_anomalous and random.random() < 0.6: # 60% chance for impossible travel
        # Simulate a location far away
        geolocation["latitude"] = random.uniform(-90, 90)
        geolocation["longitude"] = random.uniform(-180, 180)

    # Simulate safeMode toggle
    safe_mode = False
    if is_anomalous and random.random() < 0.2: # 20% chance for anomalous to be in safe mode
        safe_mode = True

    return {
        "label": label_mode,
        "timestamp": timestamp_iso, # Top-level timestamp
        "user_email": f"user_{random.randint(1, 50)}@example.com", # Simulate different users
        "context": { # New context field
            "location": geolocation,
            "safeMode": safe_mode,
            "timezone": generate_device_info(is_anomalous)['timezone'] # Ensure timezone is in context too
        },
        "data": { # This is the BehaviorDataPayload part
            "typingPatterns": generate_typing_pattern(is_anomalous),
            "mouseMovements": generate_mouse_movements(is_anomalous),
            "clickEvents": generate_click_events(is_anomalous),
            "deviceInfo": generate_device_info(is_anomalous),
            "fingerprint": generate_fingerprint() if not is_anomalous else generate_fingerprint() # Generate new FP for anomalous to simulate new device
        }
    }

# --- Script execution ---

if __name__ == "__main__":
    print(f"Generating {NUM_NORMAL_SAMPLES} normal behavior samples...")
    normal_data = [generate_behavior_sample("normal") for _ in range(NUM_NORMAL_SAMPLES)]
    
    print(f"Generating {NUM_ANOMALOUS_SAMPLES} anomalous behavior samples...")
    anomalous_data = [generate_behavior_sample("anomalous") for _ in range(NUM_ANOMALOUS_SAMPLES)]

    all_data = normal_data + anomalous_data
    random.shuffle(all_data) # Mix them up

    # Save to JSON files
    with open("behavior_normal_large.json", "w") as f:
        json.dump(normal_data, f, indent=2)
    print(f"Saved {NUM_NORMAL_SAMPLES} normal samples to behavior_normal_large.json")

    with open("behavior_anomalous_large.json", "w") as f:
        json.dump(anomalous_data, f, indent=2)
    print(f"Saved {NUM_ANOMALOUS_SAMPLES} anomalous samples to behavior_anomalous_large.json")

    # Save as a single combined file
    with open("behavior_combined_large.json", "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved {len(all_data)} combined samples to behavior_combined_large.json")

    # If you want JSONL format (one JSON object per line)
    with open("behavior_combined_large.jsonl", "w") as f:
        for entry in all_data:
            f.write(json.dumps(entry) + '\n')
    print(f"Saved {len(all_data)} combined samples to behavior_combined_large.jsonl (JSONL format)")
