"""
Configuration file for Flask application
"""
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-12345-change-in-production'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///users.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size