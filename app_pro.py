import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import io
import csv
import joblib
import numpy as np
from config import get_config

# =============================================================================
# CONFIGURATION DE L'APPLICATION
# =============================================================================

app = Flask(__name__)
app.config.from_object(get_config())
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Veuillez vous connecter pour accéder à cette page.'
login_manager.login_message_category = 'info'

# =============================================================================
# MODÈLES DE BASE DE DONNÉES (MYSQL)
# =============================================================================

class User(UserMixin, db.Model):
    """Modèle utilisateur avec MySQL"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(150), unique=True, nullable=False, index=True)
    email = db.Column(db.String(150), unique=True, nullable=False, index=True)
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    company_type = db.Column(db.String(100))
    
    predictions = db.relationship('PredictionHistory', backref='author', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    @property
    def total_predictions(self):
        return self.predictions.count()
    
    @property
    def average_demand(self):
        predictions = self.predictions.all()
        if not predictions:
            return 0
        return sum(p.predicted_demand for p in predictions) / len(predictions)


class PredictionHistory(db.Model):
    """Historique des prédictions avec MySQL"""
    __tablename__ = 'prediction_history'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Inputs
    hour = db.Column(db.Integer, nullable=False)
    temp = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, default=0.5)
    is_holiday = db.Column(db.Boolean, default=False)
    
    # Outputs
    predicted_demand = db.Column(db.Integer, nullable=False)
    segment = db.Column(db.String(100), nullable=False)
    cluster_id = db.Column(db.Integer)
    confidence_score = db.Column(db.Float, default=98.5)
    
    # Nouvelles colonnes pour les métriques ML
    model_used = db.Column(db.String(50), default='RandomForest')
    prediction_uncertainty = db.Column(db.Float)  # Écart-type des arbres
    
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    def __repr__(self):
        return f'<Prediction {self.id}: {self.predicted_demand} bikes>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'date': self.date_posted.strftime('%Y-%m-%d %H:%M:%S'),
            'hour': self.hour,
            'temperature': self.temp,
            'humidity': self.humidity,
            'is_holiday': self.is_holiday,
            'predicted_demand': self.predicted_demand,
            'segment': self.segment,
            'cluster_id': self.cluster_id,
            'confidence_score': self.confidence_score,
            'model_used': self.model_used
        }


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# =============================================================================
# MOTEUR DE PRÉDICTION IA AVEC RANDOM FOREST
# =============================================================================

class MLPredictionEngine:
    """Moteur de prédiction avec le vrai modèle Random Forest"""
    
    def __init__(self, model_path='models/rf_regressor.pkl', 
                 metadata_path='models/model_metadata.pkl',
                 classifier_path='models/rf_classifier.pkl'):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.classifier_path = classifier_path
        
        self.regressor = None
        self.classifier = None
        self.metadata = None
        self.features = None
        
        self.load_models()
        
        # Recommandations par cluster
        self.recommendations = {
            0: "📊 Standard: Maintenir l'allocation moyenne de la flotte. Monitoring régulier recommandé.",
            1: "🏖️ Loisirs: Augmenter l'approvisionnement dans les zones touristiques et parcs. Prévoir +30% de vélos.",
            2: "🚀 Pointe: Déployer du personnel pour redistribution rapide. Activez les alertes temps réel et préparez +50% de vélos.",
            3: "🔧 Faible: Période idéale pour maintenance préventive et nettoyage de la flotte. Réduisez l'allocation de 40%."
        }
    
    def load_models(self):
        """Charger les modèles ML sauvegardés"""
        try:
            if os.path.exists(self.model_path):
                self.regressor = joblib.load(self.model_path)
                print("✅ Modèle Random Forest Regressor chargé")
            else:
                print(f"⚠️ Modèle non trouvé: {self.model_path}")
                print("   Exécutez d'abord: python train_and_save_model.py")
                return
            
            if os.path.exists(self.metadata_path):
                self.metadata = joblib.load(self.metadata_path)
                self.features = self.metadata['regression_features']
                print(f"✅ Métadonnées chargées: R²={self.metadata['regression_r2_test']:.4f}")
            
            if os.path.exists(self.classifier_path):
                self.classifier = joblib.load(self.classifier_path)
                print("✅ Modèle Random Forest Classifier chargé")
                
        except Exception as e:
            print(f"❌ Erreur chargement modèles: {e}")
            self.regressor = None
    
    def prepare_input(self, hour, temp, humidity, is_holiday):
        """Préparer les features d'entrée pour le modèle"""
        
        # Créer un dictionnaire avec toutes les features
        input_data = {}
        
        # Features de base
        input_data['hr'] = hour
        input_data['temp'] = temp
        input_data['hum'] = humidity
        input_data['is_holiday'] = int(is_holiday)
        
        # Features dérivées (à adapter selon votre notebook)
        input_data['yr'] = 1  # Année (0=2011, 1=2012+)
        input_data['season'] = self._get_season(hour)
        input_data['is_peak_hour'] = 1 if hour in [7,8,9,17,18,19] else 0
        input_data['is_night'] = 1 if (hour >= 0 and hour < 6) or hour >= 22 else 0
        input_data['bad_weather'] = 1 if humidity > 0.8 else 0
        input_data['temp_windspeed_interaction'] = temp * 0.1  # Approximation
        
        # Features trigonométriques pour l'heure
        input_data['hr_sin'] = np.sin(2 * np.pi * hour / 24)
        input_data['hr_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Créer DataFrame avec les features du modèle
        input_df = pd.DataFrame([input_data])
        
        # Sélectionner uniquement les features utilisées à l'entraînement
        available_features = [f for f in self.features if f in input_df.columns]
        input_df = input_df[available_features]
        
        # Remplir les features manquantes avec 0
        for feature in self.features:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        return input_df[self.features]
    
    def predict(self, hour, temp, humidity, is_holiday):
        """Prédiction avec le modèle Random Forest"""
        
        if self.regressor is None:
            # Fallback sur l'ancien algorithme si le modèle n'est pas chargé
            return self._fallback_prediction(hour, temp, humidity, is_holiday)
        
        try:
            # Préparer les inputs
            X_input = self.prepare_input(hour, temp, humidity, is_holiday)
            
            # Prédiction avec le modèle
            prediction = self.regressor.predict(X_input)[0]
            
            # Calculer l'incertitude (écart-type des prédictions des arbres)
            tree_predictions = np.array([tree.predict(X_input)[0] 
                                        for tree in self.regressor.estimators_])
            uncertainty = np.std(tree_predictions)
            
            # Identifier le segment/cluster
            segment, cluster_id = self._identify_segment(hour, temp, humidity, is_holiday)
            
            # Calculer la confiance
            confidence = self._calculate_confidence(uncertainty)
            
            # Obtenir la recommandation
            recommendation = self.recommendations.get(cluster_id, self.recommendations[0])
            
            return {
                'demand': max(0, int(prediction)),
                'segment': segment,
                'cluster_id': cluster_id,
                'confidence': confidence,
                'recommendation': recommendation,
                'uncertainty': float(uncertainty),
                'model_metrics': {
                    'r2_score': self.metadata['regression_r2_test'],
                    'rmse': self.metadata['regression_rmse'],
                    'mae': self.metadata['regression_mae']
                }
            }
            
        except Exception as e:
            print(f"❌ Erreur prédiction: {e}")
            return self._fallback_prediction(hour, temp, humidity, is_holiday)
    
    def _get_season(self, hour):
        """Déterminer la saison approximative"""
        # Simplification: basé sur l'heure comme proxy
        if 6 <= hour <= 18:
            return 2  # Été/Printemps
        return 1  # Hiver/Automne
    
    def _identify_segment(self, hour, temp, humidity, is_holiday):
        """Identifier le segment/cluster"""
        if (7 <= hour <= 9) or (17 <= hour <= 19):
            if temp > 10 and humidity < 0.8:
                return "🚀 Haute Demande (Pointe)", 2
        
        if temp > 28 and is_holiday == 1:
            return "🏖️ Loisirs Estivaux", 1
        
        if is_holiday == 1 and temp > 20 and 10 <= hour <= 18:
            return "🏖️ Loisirs Week-end", 1
        
        if hour < 6 or hour > 22 or temp < 5 or humidity > 0.85:
            return "🔧 Demande Faible", 3
        
        return "📊 Standard", 0
    
    def _calculate_confidence(self, uncertainty):
        """Calculer le score de confiance basé sur l'incertitude"""
        # Plus l'incertitude est faible, plus la confiance est haute
        base_confidence = 95.0
        confidence_penalty = min(10.0, uncertainty / 10)
        return max(80.0, base_confidence - confidence_penalty)
    
    def _fallback_prediction(self, hour, temp, humidity, is_holiday):
        """Algorithme de secours si le modèle n'est pas chargé"""
        base_demand = 150
        adjustment = 0
        
        # Ajustements simples
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            adjustment += 200
        
        if 15 <= temp <= 25:
            adjustment += 50
        elif temp > 25:
            adjustment += 80
        
        if humidity > 0.8:
            adjustment -= 50
        
        if is_holiday:
            adjustment += 60
        
        segment, cluster_id = self._identify_segment(hour, temp, humidity, is_holiday)
        
        return {
            'demand': max(10, int(base_demand + adjustment)),
            'segment': segment,
            'cluster_id': cluster_id,
            'confidence': 85.0,
            'recommendation': self.recommendations.get(cluster_id, self.recommendations[0]),
            'uncertainty': 0.0,
            'model_metrics': None
        }


# Instance globale du moteur ML
ml_engine = MLPredictionEngine()

# =============================================================================
# ROUTES WEB
# =============================================================================

@app.route('/')
def home():
    """Page d'accueil"""
    return render_template('home.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Inscription"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        
        if not username or not email or not password:
            flash('Tous les champs sont requis.', 'danger')
            return redirect(url_for('register'))
        
        if len(password) < 8:
            flash('Le mot de passe doit contenir au moins 8 caractères.', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Cet email est déjà utilisé.', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=username).first():
            flash('Ce nom d\'utilisateur est déjà pris.', 'danger')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('🎉 Compte créé avec succès ! Connectez-vous pour commencer.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('Erreur lors de la création du compte.', 'danger')
            print(f"Erreur: {e}")
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Connexion"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            next_page = request.args.get('next')
            flash(f'Bienvenue {user.username} ! 👋', 'success')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Email ou mot de passe incorrect.', 'danger')
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """Déconnexion"""
    logout_user()
    flash('Vous avez été déconnecté avec succès.', 'success')
    return redirect(url_for('home'))


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    """Dashboard avec prédiction ML"""
    result = None
    
    if request.method == 'POST':
        try:
            hour = int(request.form.get('hour', 12))
            temp = float(request.form.get('temp', 20))
            hum = float(request.form.get('hum', 0.5))
            is_holiday = int(request.form.get('is_holiday', 0))
            
            if not (0 <= hour <= 23):
                flash('L\'heure doit être entre 0 et 23.', 'danger')
                return redirect(url_for('dashboard'))
            
            if not (0 <= hum <= 1):
                flash('L\'humidité doit être entre 0.0 et 1.0.', 'danger')
                return redirect(url_for('dashboard'))
            
            # Prédiction avec le modèle ML
            pred_result = ml_engine.predict(hour, temp, hum, is_holiday)
            
            result = {
                'pred': pred_result['demand'],
                'segment': pred_result['segment'],
                'reco': pred_result['recommendation'],
                'hour': hour,
                'temp': temp,
                'confidence': pred_result['confidence'],
                'uncertainty': pred_result['uncertainty'],
                'model_metrics': pred_result['model_metrics']
            }
            
            # Sauvegarder dans la base de données
            new_prediction = PredictionHistory(
                hour=hour, 
                temp=temp, 
                humidity=hum, 
                is_holiday=bool(is_holiday),
                predicted_demand=pred_result['demand'], 
                segment=pred_result['segment'], 
                cluster_id=pred_result['cluster_id'],
                confidence_score=pred_result['confidence'],
                prediction_uncertainty=pred_result['uncertainty'],
                model_used='RandomForest',
                author=current_user
            )
            db.session.add(new_prediction)
            db.session.commit()
            
            flash('✅ Prédiction générée avec Random Forest ML !', 'success')
            
        except ValueError as e:
            flash(f'Erreur de saisie: {e}', 'danger')
        except Exception as e:
            flash(f'Erreur lors de la prédiction: {e}', 'danger')
            print(f"Erreur: {e}")
    
    return render_template('dashboard.html', result=result)


@app.route('/history')
@login_required
def history():
    """Historique des prédictions"""
    preds = PredictionHistory.query.filter_by(user_id=current_user.id)\
        .order_by(PredictionHistory.date_posted.desc()).all()
    return render_template('history.html', preds=preds)


@app.route('/api/export/csv')
@login_required
def export_csv():
    """Export CSV"""
    preds = PredictionHistory.query.filter_by(user_id=current_user.id)\
        .order_by(PredictionHistory.date_posted.desc()).all()
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(['ID', 'Date', 'Heure', 'Température', 'Humidité',
                     'Jour Férié', 'Demande Prédite', 'Segment', 'Confiance', 
                     'Modèle', 'Incertitude'])
    
    for pred in preds:
        writer.writerow([
            pred.id, pred.date_posted.strftime('%Y-%m-%d %H:%M:%S'), pred.hour,
            pred.temp, pred.humidity, 'Oui' if pred.is_holiday else 'Non',
            pred.predicted_demand, pred.segment, f"{pred.confidence_score:.1f}%",
            pred.model_used, f"{pred.prediction_uncertainty:.2f}" if pred.prediction_uncertainty else 'N/A'
        ])
    
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8-sig')),
        mimetype='text/csv', as_attachment=True,
        download_name=f'bikepred_export_{datetime.now().strftime("%Y%m%d")}.csv'
    )


@app.route('/api/stats')
@login_required
def api_stats():
    """API stats avec métriques ML"""
    preds = current_user.predictions.all()
    
    if not preds:
        return jsonify({
            'total': 0, 
            'average_demand': 0, 
            'max_demand': 0, 
            'min_demand': 0,
            'model_info': ml_engine.metadata if ml_engine.metadata else {}
        })
    
    demands = [p.predicted_demand for p in preds]
    
    return jsonify({
        'total': len(preds),
        'average_demand': round(sum(demands) / len(demands), 1),
        'max_demand': max(demands),
        'min_demand': min(demands),
        'model_info': {
            'r2_score': ml_engine.metadata['regression_r2_test'] if ml_engine.metadata else None,
            'rmse': ml_engine.metadata['regression_rmse'] if ml_engine.metadata else None,
            'mae': ml_engine.metadata['regression_mae'] if ml_engine.metadata else None,
            'date_trained': ml_engine.metadata['date_trained'] if ml_engine.metadata else None
        }
    })


@app.errorhandler(404)
def not_found(error):
    flash('Page non trouvée.', 'danger')
    return redirect(url_for('home'))


@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    flash('Erreur serveur. Veuillez réessayer.', 'danger')
    return redirect(url_for('home'))


# =============================================================================
# INITIALISATION
# =============================================================================

def init_db():
    """Initialiser la base de données MySQL"""
    with app.app_context():
        try:
            db.create_all()
            print("✅ Tables créées avec succès dans MySQL")
            print(f"📊 Base de données: {app.config['SQLALCHEMY_DATABASE_URI']}")
        except Exception as e:
            print(f"❌ Erreur lors de la création des tables: {e}")


if __name__ == '__main__':
    init_db()
    print("\n🚀 Démarrage de BikePred Pro avec Random Forest ML")
    print(f"🌐 Accès: http://localhost:5000")
    print(f"🗄️  Base de données MySQL connectée")
    print(f"🤖 Moteur ML: {'Activé (Random Forest)' if ml_engine.regressor else 'Fallback (Règles)'}\n")
    app.run(debug=True, host='0.0.0.0', port=5000)