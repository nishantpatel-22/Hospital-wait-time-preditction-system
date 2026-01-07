from flask import Flask, request, jsonify, render_template
import pandas as pd
import sqlite3
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)
DB_NAME = "hospital_feedback.db"

# --- DATABASE SETUP ---
def init_db():
    """Initializes the SQLite database if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    # actual_waiting_time starts as NULL and is filled only after consultation
    conn.execute('''CREATE TABLE IF NOT EXISTS opd_visits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        num_patients INTEGER,
        num_doctors INTEGER,
        opd_start_hour INTEGER,
        is_weekend INTEGER,
        emergency_active INTEGER,
        predicted_waiting_time REAL,
        actual_waiting_time REAL DEFAULT NULL
    )''')
    conn.commit()
    conn.close()

# --- TRAINING PHASE ---
def get_trained_model():
    """
    Trains the ML model ONLY using records where real feedback exists.
    This ensures the model learns from ground truth, not its own predictions.
    """
    conn = sqlite3.connect(DB_NAME)
    # Only select rows where feedback has been provided
    query = "SELECT * FROM opd_visits WHERE actual_waiting_time IS NOT NULL"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Requirement: Wait until we have at least 5 real-world examples to train
    if len(df) < 5:
        return None

    X = df[["num_patients", "num_doctors", "opd_start_hour", "is_weekend", "emergency_active"]]
    y = df["actual_waiting_time"]
    
    model = LinearRegression()
    model.fit(X, y)
    return model

@app.route('/')
def index():
    return render_template('index.html')

# --- PREDICTION PHASE ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    1. Receives inputs from Frontend.
    2. Generates a prediction (using ML or base formula).
    3. Saves the record with actual_waiting_time as NULL.
    """
    data = request.json
    num_patients = int(data['num_patients'])
    num_doctors = int(data['num_doctors'])
    
    # Try to get a model trained on REAL feedback
    model = get_trained_model()
    
    if model:
        # Use ML to predict
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]
    else:
        # Fallback: Simple math if not enough real data exists yet
        prediction = (num_patients / num_doctors) * 2 

    # Add rule-based buffers
    if int(data['emergency_active']) == 1: 
        prediction += 20
    if num_patients > 100:
        prediction += 10
        
    prediction = round(prediction, 2)

    # Save intent to database so we can update it later with feedback
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO opd_visits 
        (num_patients, num_doctors, opd_start_hour, is_weekend, emergency_active, predicted_waiting_time) 
        VALUES (?, ?, ?, ?, ?, ?)''',
        (num_patients, num_doctors, data['opd_start_hour'], data['is_weekend'], 
         data['emergency_active'], prediction))
    
    visit_id = cursor.lastrowid # We send this ID to the frontend to link feedback
    conn.commit()
    conn.close()

    return jsonify({"prediction": prediction, "visit_id": visit_id})

# --- FEEDBACK PHASE ---
@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Updates a specific record with the real time taken.
    The next prediction will include this data in its training set.
    """
    data = request.json
    visit_id = data['visit_id']
    actual_time = float(data['actual_waiting_time'])

    conn = sqlite3.connect(DB_NAME)
    conn.execute("UPDATE opd_visits SET actual_waiting_time = ? WHERE id = ?", 
                 (actual_time, visit_id))
    conn.commit()
    conn.close()

    return jsonify({"status": "success", "message": "Feedback recorded. Model updated."})

if __name__ == '__main__':
    init_db()
    app.run(debug=True)