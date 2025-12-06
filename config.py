import os
from datetime import timedelta

class Config:
    """Configuration de base pour l'application"""
    
    # Clé secrète pour les sessions
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-change-in-production-2025'
    
    # Configuration SQLAlchemy
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)
    
    # Configuration session
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'


class DevelopmentConfig(Config):
    """Configuration pour le développement (XAMPP/MySQL)"""
    
    DEBUG = True
    TESTING = False
    
    # ========================================
    # CONNEXION MYSQL (XAMPP)
    # ========================================
    # Format: mysql+pymysql://utilisateur:motdepasse@hote:port/nom_base
    
    MYSQL_USER = 'root'  # Par défaut dans XAMPP
    MYSQL_PASSWORD = ''  # Vide par défaut dans XAMPP
    MYSQL_HOST = 'localhost'  # ou '127.0.0.1'
    MYSQL_PORT = 3306
    MYSQL_DATABASE = 'bikepred_prod'  # Nom de votre base de données
    
    # URI de connexion MySQL
    SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
    
    # Options du moteur de base de données
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,  # Vérifier la connexion avant utilisation
        'pool_recycle': 3600,   # Recycler les connexions après 1h
        'pool_size': 10,        # Nombre de connexions dans le pool
        'max_overflow': 20,     # Connexions supplémentaires en cas de besoin
        'echo': False           # Mettre True pour voir les requêtes SQL dans la console
    }


class ProductionConfig(Config):
    """Configuration pour la production"""
    
    DEBUG = False
    TESTING = False
    
    # En production, utiliser des variables d'environnement
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'mysql+pymysql://root@localhost/bikepred_prod'
    
    # Sécurité renforcée
    SESSION_COOKIE_SECURE = True  # HTTPS uniquement
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 3600,
        'pool_size': 15,
        'max_overflow': 30
    }


# Dictionnaire de configuration
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}


def get_config():
    """Retourner la configuration selon l'environnement"""
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])