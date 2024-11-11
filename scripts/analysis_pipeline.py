import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# File paths
raw_data_path = 'data/CCG.csv'
processed_data_path = 'data/cleaned_fan_data.csv'
model_path = 'models/attendance_predictive_model.pkl'

# Step 1: Data Cleaning and Preprocessing
def data_cleaning(raw_data_path):
    print("Step 1: Data Cleaning and Preprocessing")
    # Load raw data
    data = pd.read_excel(raw_data_path)
    
    # Handle missing values and preprocess columns
    data.drop_duplicates(inplace=True)
    data.fillna({'Income_Level': 'Unknown', 'Fan_Type': 'Unknown'}, inplace=True)
    data['Income_Level'] = data['Income_Level'].map({'Low': 1, 'Medium': 2, 'High': 3, 'Unknown': 0})
    data['STM_Holder'] = data['STM_Holder'].map({0: 'No', 1: 'Yes'})
    data.dropna(inplace=True)  # Drop any remaining NaNs
    
    # Save cleaned data
    data.to_csv(processed_data_path, index=False)
    print(f"Cleaned data saved to {processed_data_path}\n")
    return data

# Step 2: Clustering Analysis
def clustering_analysis(data):
    print("Step 2: Clustering Analysis")
    features = ['Income_Level', 'Lifetime_Ticket_Units', 'Lifetime_Concessions_Spend', 'Distance_to_Arena_Miles']
    
    # Normalize features
    X = data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X_scaled)
    print("Clustering analysis completed.\n")
    return data

# Step 3: Predictive Modeling
def predictive_modeling(data):
    print("Step 3: Predictive Modeling")
    features = ['Income_Level', 'Lifetime_Ticket_Units', 'Lifetime_Concessions_Spend', 'Distance_to_Arena_Miles', 'Cluster']
    target = (data['Attendance_Trend'] == 'Increasing').astype(int)  # Binary target: Increasing attendance or not
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data[features], target, test_size=0.2, random_state=42)
    
    # Train a Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}\n")
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))
    
    # Save the trained model
    joblib.dump(model, model_path)
    print(f"Trained model saved to {model_path}\n")

# Step 4: Execute Analysis Pipeline
def run_analysis_pipeline():
    # Step 1: Clean Data
    cleaned_data = data_cleaning(raw_data_path)
    
    # Step 2: Perform Clustering Analysis
    clustered_data = clustering_analysis(cleaned_data)
    
    # Step 3: Perform Predictive Modeling
    predictive_modeling(clustered_data)

if __name__ == "__main__":
    run_analysis_pipeline()
