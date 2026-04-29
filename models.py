"""
Database models for user authentication and prediction history
"""
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """
    User model for authentication
    """
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationship with predictions
    predictions = db.relationship('PredictionHistory', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class PredictionHistory(db.Model):
    """
    Model to store prediction history for each user
    """
    __tablename__ = 'prediction_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Input parameters
    voltage = db.Column(db.Float, nullable=False)
    current = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    
    # Predictions from different models
    rf_prediction = db.Column(db.Float, nullable=False)
    ann_prediction = db.Column(db.Float, nullable=False)
    ensemble_prediction = db.Column(db.Float, nullable=True)
    
    # Metrics
    variance = db.Column(db.Float, nullable=True)
    std_deviation = db.Column(db.Float, nullable=True)
    rf_mse = db.Column(db.Float, nullable=True)
    ann_mse = db.Column(db.Float, nullable=True)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(50), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    
    def to_dict(self):
        """Convert to dictionary for JSON responses"""
        return {
            'id': self.id,
            'voltage': self.voltage,
            'current': self.current,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'rf_prediction': self.rf_prediction,
            'ann_prediction': self.ann_prediction,
            'ensemble_prediction': self.ensemble_prediction,
            'variance': self.variance,
            'std_deviation': self.std_deviation,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None
        }
    
    def __repr__(self):
        return f'<Prediction {self.id} by User {self.user_id}>'

def init_db(app):
    """Initialize database with app context"""
    with app.app_context():
        db.create_all()
        print("✓ Database initialized")