from flask import Flask, request, jsonify, render_template
import pandas as pd
import sqlite3
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

# --- ROBUST PATH SETUP ---
# This ensures the database file is created in the correct folder on Render
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_NAME = os.path.join(BASE_DIR, "hospital_feedback.db")

# --- DATABASE SETUP ---
def init_db():
    """Initializes the SQLite database with all necessary columns."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # We create the table with ALL columns needed for prediction and feedback
    cursor.execute('''CREATE TABLE IF NOT EXISTS opd_visits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        num_patients INTEGER,
        num_doctors INTEGER,
        opd_start_hour INTEGER,
        is_weekend INTEGER,
        emergency_active INTEGER DEFAULT 0,
        predicted_waiting_time REAL,
        actual_waiting_time REAL DEFAULT NULL
    )''')
    
    # Check if we need to insert sample data for the prediction to work immediately
    cursor.execute("SELECT COUNT(*) FROM opd_visits")
    if cursor.fetchone()[0] == 0:
        cursor.execute('''INSERT INTO opd_visits 
            (num_patients, num_doctors, opd_start_hour, is_weekend, emergency_active, predicted_waiting_time) 
            VALUES (10, 2, 9, 0, 0, 10.0)''')
    
    conn.commit()
    conn.close()

# Initialize the database immediately when the app starts
init_db()

# --- TRAINING PHASE ---
def get_trained_model():
    """Trains the ML model ONLY using records where real feedback exists."""
    try:
        conn = sqlite3.connect(DB_NAME)
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
    except Exception as e:
        print(f"Training error: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

# --- PREDICTION PHASE ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Extract values with defaults to prevent crashes if a key is missing
    num_patients = int(data.get('num_patients', 0))
    num_doctors = int(data.get('num_doctors', 1)) # Prevent division by zero
    opd_start_hour = int(data.get('opd_start_hour', 9))
    is_weekend = int(data.get('is_weekend', 0))
    emergency_active = int(data.get('emergency_active', 0))
    
    # Try to get a model trained on REAL feedback
    model = get_trained_model()
    
    if model:
        # Prepare data for ML model
        input_data = pd.DataFrame([[num_patients, num_doctors, opd_start_hour, is_weekend, emergency_active]], 
                                   columns=["num_patients", "num_doctors", "opd_start_hour", "is_weekend", "emergency_active"])
        prediction = model.predict(input_data)[0]
    else:
        # Fallback: Simple formula
        prediction = (num_patients / num_doctors) * 2 

    # Add rule-based buffers
    if emergency_active == 1: 
        prediction += 20
    if num_patients > 100:
        prediction += 10
        
    prediction = max(0, round(prediction, 2)) # Ensure no negative wait times

    # Save to database
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO opd_visits 
        (num_patients, num_doctors, opd_start_hour, is_weekend, emergency_active, predicted_waiting_time) 
        VALUES (?, ?, ?, ?, ?, ?)''',
        (num_patients, num_doctors, opd_start_hour, is_weekend, emergency_active, prediction))
    
    visit_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return jsonify({"prediction": prediction, "visit_id": visit_id})

# --- FEEDBACK PHASE ---
@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    visit_id = data.get('visit_id')
    actual_time = float(data.get('actual_waiting_time', 0))

    conn = sqlite3.connect(DB_NAME)
    conn.execute("UPDATE opd_visits SET actual_waiting_time = ? WHERE id = ?", 
                 (actual_time, visit_id))
    conn.commit()
    conn.close()

    return jsonify({"status": "success", "message": "Feedback recorded."})

if __name__ == '__main__':
    app.run(debug=True)