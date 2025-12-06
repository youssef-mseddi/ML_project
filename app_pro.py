import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import io
import csv
from config import get_config

# =============================================================================
# CONFIGURATION DE L'APPLICATION
# =============================================================================

app = Flask(__name__)

# Charger la configuration depuis config.py
app.config.from_object(get_config())

# Initialiser SQLAlchemy avec MySQL
db = SQLAlchemy(app)

# Configuration Flask-Login
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
    password = db.Column(db.String(255), nullable=False)  # 255 pour le hash
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    company_type = db.Column(db.String(100))
    
    # Relations
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
    
    # Metadata
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    def __repr__(self):
        return f'<Prediction {self.id}: {self.predicted_demand} bikes>'
    
    def to_dict(self):
        """Convertir en dictionnaire pour export"""
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
            'confidence_score': self.confidence_score
        }


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# =============================================================================
# MOTEUR DE PRÉDICTION IA
# =============================================================================

class PredictionEngine:
    """Moteur de prédiction avec Random Forest simulation"""
    
    def __init__(self, data_path='bike_sharing_final_with_clusters.csv'):
        self.data_path = data_path
        self.df = None
        self.load_data()
        
        self.recommendations = {
            0: "📊 Standard: Maintenir l'allocation moyenne de la flotte. Monitoring régulier recommandé.",
            1: "🏖️ Loisirs: Augmenter l'approvisionnement dans les zones touristiques et parcs. Prévoir +30% de vélos.",
            2: "🚀 Pointe: Déployer du personnel pour redistribution rapide. Activez les alertes temps réel et préparez +50% de vélos.",
            3: "🔧 Faible: Période idéale pour maintenance préventive et nettoyage de la flotte. Réduisez l'allocation de 40%."
        }
    
    def load_data(self):
        """Chargement des données d'entraînement"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Données chargées: {len(self.df)} enregistrements")
        except Exception as e:
            print(f"⚠️ Erreur chargement données: {e}")
            self.df = None
    
    def predict(self, hour, temp, humidity, is_holiday):
        """Prédiction de la demande avec segmentation intelligente"""
        
        # 1. Segmentation
        segment, cluster_id = self._identify_segment(hour, temp, humidity, is_holiday)
        
        # 2. Demande baseline
        if self.df is not None and 'cluster' in self.df.columns:
            base_demand = self.df[self.df['cluster'] == cluster_id]['cnt'].mean()
        elif self.df is not None:
            base_demand = self.df['cnt'].mean()
        else:
            base_demand = 150
        
        if pd.isna(base_demand):
            base_demand = 150
        
        # 3. Ajustements
        adjustments = self._calculate_adjustments(hour, temp, humidity, is_holiday, cluster_id)
        
        # 4. Calcul final
        predicted_demand = int(max(10, base_demand + adjustments))
        
        # 5. Confiance
        confidence = self._calculate_confidence(hour, temp, humidity)
        
        return predicted_demand, segment, cluster_id, confidence
    
    def _identify_segment(self, hour, temp, humidity, is_holiday):
        """Identification du segment"""
        if (7 <= hour <= 9) or (17 <= hour <= 19):
            if temp > 10 and humidity < 0.8:
                return "🚀 Haute Demande (Pointe)", 2
        
        if temp > 28 and is_holiday == 1:
            return "🏖️ Loisirs Estivaux", 1
        
        if is_holiday == 1 and temp > 20 and 10 <= hour <= 18:
            return "🏖️ Loisirs Week-end", 1
        
        if hour < 6 or hour > 22 or temp < 5 or humidity > 0.85:
            return "🔧 Demande Faible", 3
        
        if temp < 0 or humidity > 0.9:
            return "🔧 Conditions Défavorables", 3
        
        return "📊 Standard", 0
    
    def _calculate_adjustments(self, hour, temp, humidity, is_holiday, cluster_id):
        """Calcul des ajustements"""
        adjustment = 0
        cluster_boosts = {2: 450, 1: 200, 0: 0, 3: -120}
        adjustment += cluster_boosts.get(cluster_id, 0)
        
        hour_factor = 15 * (hour - 12) if 6 <= hour <= 22 else -50
        adjustment += hour_factor
        
        if 15 <= temp <= 25:
            adjustment += 8 * (temp - 20)
        elif temp > 25:
            adjustment += 40 + 3 * (temp - 25)
        else:
            adjustment += 5 * (temp - 15)
        
        humidity_penalty = -150 * abs(humidity - 0.5)
        adjustment += humidity_penalty
        
        if is_holiday == 1:
            adjustment += 80
        
        return adjustment
    
    def _calculate_confidence(self, hour, temp, humidity):
        """Score de confiance"""
        confidence = 98.5
        if temp < -5 or temp > 40:
            confidence -= 5
        if humidity > 0.95 or humidity < 0.1:
            confidence -= 3
        if hour < 5 or hour > 23:
            confidence -= 2
        return max(85.0, confidence)
    
    def get_recommendation(self, cluster_id):
        """Recommandation"""
        return self.recommendations.get(cluster_id, self.recommendations[0])


# Instance globale
prediction_engine = PredictionEngine()

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
    """Dashboard"""
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
            
            pred, segment, cluster_id, confidence = prediction_engine.predict(hour, temp, hum, is_holiday)
            reco = prediction_engine.get_recommendation(cluster_id)
            
            result = {
                'pred': pred,
                'segment': segment,
                'reco': reco,
                'hour': hour,
                'temp': temp,
                'confidence': confidence
            }
            
            new_prediction = PredictionHistory(
                hour=hour, temp=temp, humidity=hum, is_holiday=bool(is_holiday),
                predicted_demand=pred, segment=segment, cluster_id=cluster_id,
                confidence_score=confidence, author=current_user
            )
            db.session.add(new_prediction)
            db.session.commit()
            
            flash('✅ Prédiction générée avec succès !', 'success')
            
        except ValueError as e:
            flash(f'Erreur de saisie: {e}', 'danger')
        except Exception as e:
            flash(f'Erreur lors de la prédiction: {e}', 'danger')
            print(f"Erreur: {e}")
    
    return render_template('dashboard.html', result=result)


@app.route('/history')
@login_required
def history():
    """Historique"""
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
                     'Jour Férié', 'Demande Prédite', 'Segment', 'Confiance'])
    
    for pred in preds:
        writer.writerow([
            pred.id, pred.date_posted.strftime('%Y-%m-%d %H:%M:%S'), pred.hour,
            pred.temp, pred.humidity, 'Oui' if pred.is_holiday else 'Non',
            pred.predicted_demand, pred.segment, f"{pred.confidence_score:.1f}%"
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
    """API stats"""
    preds = current_user.predictions.all()
    
    if not preds:
        return jsonify({'total': 0, 'average_demand': 0, 'max_demand': 0, 'min_demand': 0})
    
    demands = [p.predicted_demand for p in preds]
    
    return jsonify({
        'total': len(preds),
        'average_demand': round(sum(demands) / len(demands), 1),
        'max_demand': max(demands),
        'min_demand': min(demands)
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
            # Créer toutes les tables
            db.create_all()
            print("✅ Tables créées avec succès dans MySQL")
            print(f"📊 Base de données: {app.config['SQLALCHEMY_DATABASE_URI']}")
        except Exception as e:
            print(f"❌ Erreur lors de la création des tables: {e}")
            print("Vérifiez que MySQL est démarré et que la base de données existe")


if __name__ == '__main__':
    init_db()
    print("\n🚀 Démarrage de l'application BikePred Pro")
    print(f"🌐 Accès: http://localhost:5000")
    print(f"🗄️  Base de données MySQL connectée\n")
    app.run(debug=True, host='0.0.0.0', port=5000)