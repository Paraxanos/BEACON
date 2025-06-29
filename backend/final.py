import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
import numpy as np
import joblib # For saving the model
import matplotlib.pyplot as plt # Still needed if plots were generated again in this run

# Load the processed DataFrame
try:
    df = pd.read_csv('processed_behavior_features.csv')
    print("Loaded 'processed_behavior_features.csv'.")
except FileNotFoundError:
    print("Error: 'processed_behavior_features.csv' not found.")
    print("Please ensure the file is in the same directory or run the feature engineering script first.")
    exit()

# Separate features (X) and labels (y)
X = df.drop('label', axis=1)
y = df['label']

# Identify numerical columns for scaling
numerical_cols_to_scale = [col for col in X.columns if X[col].dtype in ['float64', 'int64'] and not (X[col].isin([0, 1]).all() and X[col].nunique() <= 2)]

# Apply scaling only to the identified numerical columns
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numerical_cols_to_scale] = scaler.fit_transform(X[numerical_cols_to_scale])

# Train-Test Split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Re-train the model (or load if preferred, but re-training ensures consistency with the code flow)
contamination_rate = y_train.value_counts(normalize=True)[1]
model = IsolationForest(contamination=contamination_rate, random_state=42, n_estimators=100)
model.fit(X_train)

# Predict on test set
y_test_pred = model.predict(X_test)
y_test_pred_mapped = np.where(y_test_pred == -1, 1, 0) # Map -1/1 to 0/1 for comparison

print("\n--- Further Analysis: Inspecting False Positives ---")

# Find False Positives (Actual 0, Predicted 1)
false_positives_indices = X_test[(y_test == 0) & (y_test_pred_mapped == 1)].index
false_positives_count = len(false_positives_indices)

if false_positives_count > 0:
    print(f"Found {false_positives_count} False Positive(s).")
    # Display details of false positives
    for i, idx in enumerate(false_positives_indices):
        original_index_in_df = df.index[df.index == idx].tolist() # Get original index from the full dataframe if needed
        print(f"\nFalse Positive {i+1} (Original df index: {original_index_in_df}):")
        print("Actual Label: 0 (Normal)")
        print("Predicted Label: 1 (Anomalous)")
        print("Features:")
        print(X_test.loc[idx])
        # Optionally, print the anomaly score for this specific instance
        anomaly_score = model.decision_function(X_test.loc[[idx]])[0]
        print(f"Anomaly Score: {anomaly_score:.4f}")
else:
    print("No False Positives found in the test set.")

print("\n--- Saving the Best Model ---")

# Define the filename for the saved model
model_filename = 'isolation_forest_model.joblib'

# Save the trained model
joblib.dump(model, model_filename)
print(f"Model successfully saved to '{model_filename}'")

# You can load the model later using:
# loaded_model = joblib.load(model_filename)
# print(f"Model loaded from '{model_filename}'")

print("\nFurther analysis and model saving complete. The model is now ready for deployment!")