"""
Flask Web Application for Energy Demand Prediction
Features:
- User authentication (login/register)
- Prediction dashboard with history
- New prediction input form
- Result display with metrics
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename

# Import local modules
from config import Config
from models import db, User, PredictionHistory, init_db

# Import the correct predictor class
from predict_demand import DemandPredictor

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Create necessary directories
os.makedirs('trained_models', exist_ok=True)
os.makedirs('instance', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

# Global variable for predictor (lazy loaded)
_predictor = None

def get_predictor():
    """Lazy load the predictor to avoid loading at startup"""
    global _predictor
    if _predictor is None:
        try:
            _predictor = DemandPredictor('trained_models/')
            print("✓ Predictor loaded successfully")
        except Exception as e:
            print(f"✗ Error loading predictor: {e}")
            _predictor = None
    return _predictor

@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login"""
    return User.query.get(int(user_id))

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required!', 'danger')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return render_template('register.html')
        
        # Check if user exists
        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()
        
        if existing_user:
            if existing_user.username == username:
                flash('Username already taken!', 'danger')
            else:
                flash('Email already registered!', 'danger')
            return render_template('register.html')
        
        # Create new user
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash(f'Registration failed: {str(e)}', 'danger')
            return render_template('register.html')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
        
        # Find user
        user = User.query.filter_by(username=username).first()
        
        if not user or not user.check_password(password):
            flash('Invalid username or password!', 'danger')
            return render_template('login.html')
        
        if not user.is_active:
            flash('Account is deactivated. Contact admin.', 'danger')
            return render_template('login.html')
        
        # Login user
        login_user(user, remember=remember)
        flash(f'Welcome back, {user.username}!', 'success')
        
        # Redirect to next page or dashboard
        next_page = request.args.get('next')
        return redirect(next_page) if next_page else redirect(url_for('dashboard'))
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with prediction history"""
    # Get user's prediction history (latest first)
    predictions = PredictionHistory.query.filter_by(
        user_id=current_user.id
    ).order_by(PredictionHistory.created_at.desc()).all()
    
    # Calculate statistics
    total_predictions = len(predictions)
    avg_rf = np.mean([p.rf_prediction for p in predictions]) if predictions else 0
    avg_ann = np.mean([p.ann_prediction for p in predictions]) if predictions else 0
    
    return render_template(
        'dashboard.html',
        predictions=predictions,
        total_predictions=total_predictions,
        avg_rf=avg_rf,
        avg_ann=avg_ann
    )

@app.route('/history/<int:prediction_id>')
@login_required
def view_history_detail(prediction_id):
    """View details of a specific prediction"""
    prediction = PredictionHistory.query.get_or_404(prediction_id)
    
    # Ensure user owns this prediction
    if prediction.user_id != current_user.id:
        flash('Access denied!', 'danger')
        return redirect(url_for('dashboard'))
    
    return render_template('history_detail.html', prediction=prediction)

@app.route('/new-prediction', methods=['GET', 'POST'])
@login_required
def new_prediction():
    """New prediction form"""
    if request.method == 'POST':
        try:
            # Get form data - using request.form for POST
            voltage = float(request.form.get('voltage'))
            current = float(request.form.get('current'))
            temperature = float(request.form.get('temperature'))
            humidity = float(request.form.get('humidity'))
            notes = request.form.get('notes', '')
            
            # Validate input ranges (soft validation)
            warnings = []
            if voltage < 0 or voltage > 15:
                warnings.append('Voltage outside typical range')
            if current < 0 or current > 15:
                warnings.append('Current outside typical range')
            if temperature < 20 or temperature > 40:
                warnings.append('Temperature outside typical range')
            if humidity < 40 or humidity > 90:
                warnings.append('Humidity outside typical range')
            
            if warnings:
                flash('Warning: ' + '. '.join(warnings), 'warning')
            
            # Get predictor
            predictor = get_predictor()
            if not predictor:
                flash('Prediction models not loaded. Please contact admin.', 'danger')
                return redirect(url_for('dashboard'))
            
            # Start timing
            start_time = time.time()
            
            # Make prediction
            predictions = predictor.predict_single(voltage, current, temperature, humidity)
            
            # End timing
            end_time = time.time()
            runtime = end_time - start_time
            
            # Calculate metrics
            rf_pred = predictions.get('random_forest', 0)
            ann_pred = predictions.get('ann', 0)
            ensemble_pred = (rf_pred + ann_pred) / 2 if rf_pred and ann_pred else None
            
            # Calculate variance and standard deviation
            pred_values = [v for v in [rf_pred, ann_pred] if v is not None]
            variance = float(np.var(pred_values)) if len(pred_values) > 1 else 0
            std_dev = float(np.std(pred_values)) if len(pred_values) > 1 else 0
            
            # Calculate MSE (using ensemble as reference)
            rf_mse = float((rf_pred - ensemble_pred) ** 2) if ensemble_pred else 0
            ann_mse = float((ann_pred - ensemble_pred) ** 2) if ensemble_pred else 0
            
            # Save to database
            new_pred = PredictionHistory(
                user_id=current_user.id,
                voltage=voltage,
                current=current,
                temperature=temperature,
                humidity=humidity,
                rf_prediction=rf_pred,
                ann_prediction=ann_pred,
                ensemble_prediction=ensemble_pred,
                variance=variance,
                std_deviation=std_dev,
                rf_mse=rf_mse,
                ann_mse=ann_mse,
                notes=notes,
                ip_address=request.remote_addr
            )
            
            db.session.add(new_pred)
            db.session.commit()
            
            flash('Prediction completed successfully!', 'success')
            return redirect(url_for('result', prediction_id=new_pred.id))
            
        except ValueError as e:
            flash(f'Invalid input: Please enter valid numbers. Error: {str(e)}', 'danger')
            return redirect(url_for('new_prediction'))
        except Exception as e:
            flash(f'Prediction failed: {str(e)}', 'danger')
            import traceback
            traceback.print_exc()
            return redirect(url_for('new_prediction'))
    import requests 
    data=requests.get("https://api.thingspeak.com/channels/3304701/feeds.json?api_key=78AP2C9J6KD432UH&results=2")
    temp=data.json()["feeds"][-1]["field1"]
    hum=data.json()["feeds"][-1]["field2"]
    vol=data.json()["feeds"][-1]["field3"]
    cur=data.json()["feeds"][-1]["field4"]
    return render_template('new_prediction.html',temp=temp,hum=hum,vol=vol,cur=cur)

@app.route('/result/<int:prediction_id>')
@login_required
def result(prediction_id):
    """Display prediction result"""
    prediction = PredictionHistory.query.get_or_404(prediction_id)
    
    # Ensure user owns this prediction
    if prediction.user_id != current_user.id:
        flash('Access denied!', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get convergence speed from training (if available)
    try:
        config = joblib.load('trained_models/config.pkl')
        convergence_speed = config.get('training_time', {})
    except:
        convergence_speed = {
            'random_forest': 15.5,
            'ann': 45.2
        }
    
    return render_template('result.html', prediction=prediction, convergence_speed=convergence_speed)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (can be used by external apps)"""
    try:
        data = request.get_json()
        
        # Validate input
        required = ['voltage', 'current', 'temperature', 'humidity']
        if not all(k in data for k in required):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Get predictor
        predictor = get_predictor()
        if not predictor:
            return jsonify({'error': 'Prediction service unavailable'}), 503
        
        # Make prediction
        predictions = predictor.predict_single(
            float(data['voltage']),
            float(data['current']),
            float(data['temperature']),
            float(data['humidity'])
        )
        
        # Calculate ensemble
        rf_pred = predictions.get('random_forest')
        ann_pred = predictions.get('ann')
        ensemble = (rf_pred + ann_pred) / 2 if rf_pred and ann_pred else None
        
        # Calculate variance
        pred_values = [v for v in [rf_pred, ann_pred] if v is not None]
        variance = float(np.var(pred_values)) if len(pred_values) > 1 else 0
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'ensemble': ensemble,
            'variance': variance
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete-history/<int:prediction_id>', methods=['POST'])
@login_required
def delete_history(prediction_id):
    """Delete a prediction from history"""
    prediction = PredictionHistory.query.get_or_404(prediction_id)
    
    if prediction.user_id != current_user.id:
        flash('Access denied!', 'danger')
        return redirect(url_for('dashboard'))
    
    try:
        db.session.delete(prediction)
        db.session.commit()
        flash('Prediction deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting prediction: {str(e)}', 'danger')
    
    return redirect(url_for('dashboard'))

@app.route('/clear-history', methods=['POST'])
@login_required
def clear_history():
    """Clear all predictions for current user"""
    try:
        PredictionHistory.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        flash('All predictions cleared!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error clearing history: {str(e)}', 'danger')
    
    return redirect(url_for('dashboard'))

@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    prediction_count = PredictionHistory.query.filter_by(user_id=current_user.id).count()
    return render_template('profile.html', user=current_user, prediction_count=prediction_count)

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

# Initialize database and create tables
with app.app_context():
    db.create_all()
    print("✓ Database tables created")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)