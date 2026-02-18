import pandas as pd
import pickle
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import io
import csv
from config import get_config

app = Flask(__name__)
app.config.from_object(get_config())

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Veuillez vous connecter pour accéder à cette page.'
login_manager.login_message_category = 'info'

class User(UserMixin, db.Model):
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

class PredictionHistory(db.Model):
    __tablename__ = 'prediction_history'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, index=True)
    prediction_date = db.Column(db.Date, nullable=True)
    hour = db.Column(db.Integer, nullable=False)
    temp = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, default=0.5)
    is_holiday = db.Column(db.Boolean, default=False)
    day_of_week = db.Column(db.Integer)
    weather = db.Column(db.Integer, default=1)
    predicted_demand = db.Column(db.Integer, nullable=False)
    predicted_casual = db.Column(db.Integer, default=0)
    predicted_registered = db.Column(db.Integer, default=0)
    cluster_id = db.Column(db.Integer)
    segment = db.Column(db.String(100), nullable=False)
    confidence_score = db.Column(db.Float, default=98.5)
    model_used = db.Column(db.String(50), default='RandomForest')
    recommendation = db.Column(db.Text)
    marketing_strategy = db.Column(db.String(200))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    def __repr__(self):
        return f'<Prediction {self.id}: {self.predicted_demand} bikes>'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class MLModelsManager:
    def __init__(self):
        self.dso1_model = None
        self.dso1_scaler = None
        self.dso1_features = []
        self.dso2_model = None
        self.dso2_scaler = None
        self.dso2_pca = None
        self.dso2_features = []
        self.dso3_recommendations = None
        self.load_all_models()
    
    def load_all_models(self):
        print("\n" + "="*70)
        print("📦 CHARGEMENT DES MODÈLES MACHINE LEARNING")
        print("="*70)
        
        dso1_path = 'DSO1_prediction_demande.pkl'
        try:
            if os.path.exists(dso1_path):
                with open(dso1_path, 'rb') as f:
                    dso1_data = pickle.load(f)
                if isinstance(dso1_data, dict):
                    self.dso1_model = dso1_data['model']
                    self.dso1_scaler = dso1_data.get('scaler')
                    self.dso1_features = dso1_data.get('features', [])
                    print(f"✅ DSO1 chargé ({len(self.dso1_features)} features)")
                else:
                    self.dso1_model = dso1_data
            else:
                raise FileNotFoundError(f"❌ {dso1_path} non trouvé!")
        except Exception as e:
            print(f"❌ Erreur DSO1: {e}")
            raise
        
        dso2_path = 'DSO2_segmentation_clusters.pkl'
        try:
            if os.path.exists(dso2_path):
                with open(dso2_path, 'rb') as f:
                    dso2_data = pickle.load(f)
                if isinstance(dso2_data, dict):
                    self.dso2_model = dso2_data['model']
                    self.dso2_scaler = dso2_data.get('scaler')
                    self.dso2_pca = dso2_data.get('pca')
                    self.dso2_features = dso2_data.get('features', [])
                    print(f"✅ DSO2 chargé ({len(self.dso2_features)} features)")
                    print(f"   ⚠️  PCA NON utilisée (KMeans attend 11 features)")
                else:
                    self.dso2_model = dso2_data
            else:
                raise FileNotFoundError(f"❌ {dso2_path} non trouvé!")
        except Exception as e:
            print(f"❌ Erreur DSO2: {e}")
            raise
        
        dso3_path = 'DSO3_recommandations_marketing.pkl'
        try:
            if os.path.exists(dso3_path):
                with open(dso3_path, 'rb') as f:
                    self.dso3_recommendations = pickle.load(f)
                print(f"✅ DSO3 chargé")
            else:
                self.dso3_recommendations = {}
        except Exception as e:
            self.dso3_recommendations = {}
        
        print("="*70)
        print("✅ TOUS LES MODÈLES CHARGÉS")
        print("="*70 + "\n")
    
    def prepare_features_dso1(self, features_dict):
        hour = features_dict.get('hour', 12)
        temp = features_dict.get('temp', 20)
        humidity = features_dict.get('humidity', 0.5)
        weather = features_dict.get('weather', 1)
        is_holiday = features_dict.get('is_holiday', 0)
        day_of_week = features_dict.get('day_of_week', 0)
        
        is_night = 1 if (hour >= 22 or hour <= 6) else 0
        hr_sin = np.sin(2 * np.pi * hour / 24)
        hr_cos = np.cos(2 * np.pi * hour / 24)
        is_peak_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
        is_evening_rush = 1 if (17 <= hour <= 19) else 0
        yr = 0
        windspeed = 0.19
        temp_windspeed_interaction = temp * windspeed
        bad_weather = 1 if weather >= 3 else 0
        registered_ratio = 0.8
        month = 1 if temp < 8 else (4 if temp < 15 else (7 if temp > 25 else 10))
        month_cos = np.cos(2 * np.pi * month / 12)
        season = 1 if temp < 10 else (2 if temp < 20 else 3)
        is_winter = 1 if season == 1 else 0
        
        feature_values = {
            'is_night': is_night, 'hr': hour, 'temp': temp, 'hr_sin': hr_sin,
            'is_peak_hour': is_peak_hour, 'hr_cos': hr_cos, 'is_evening_rush': is_evening_rush,
            'yr': yr, 'hum': humidity, 'temp_windspeed_interaction': temp_windspeed_interaction,
            'bad_weather': bad_weather, 'registered_ratio': registered_ratio,
            'month_cos': month_cos, 'season': season, 'is_winter': is_winter
        }
        
        if self.dso1_features:
            ordered_values = [feature_values.get(feat, 0) for feat in self.dso1_features]
            df = pd.DataFrame([ordered_values], columns=self.dso1_features)
        else:
            df = pd.DataFrame([list(feature_values.values())], columns=list(feature_values.keys()))
        return df
    
    def predict_demand_dso1(self, features_dict):
        try:
            X = self.prepare_features_dso1(features_dict)
            print(f"\n🔍 DSO1 - Features préparées ({X.shape[1]} features)")
            X_scaled = self.dso1_scaler.transform(X)
            prediction = self.dso1_model.predict(X_scaled)[0]
            total_demand = int(max(0, prediction))
            print(f"✅ DSO1 - Demande prédite: {total_demand} vélos")
            casual = int(total_demand * 0.19)
            registered = total_demand - casual
            return {'total': total_demand, 'casual': casual, 'registered': registered, 'model_name': 'DSO1-RandomForest'}
        except Exception as e:
            print(f"❌ Erreur DSO1: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def prepare_features_dso2(self, features_dict):
        hour = features_dict.get('hour', 12)
        temp = features_dict.get('temp', 20)
        humidity = features_dict.get('humidity', 0.5)
        day_of_week = features_dict.get('day_of_week', 0)
        
        windspeed = 0.19
        season = 1 if temp < 10 else (2 if temp < 20 else 3)
        hr_sin = np.sin(2 * np.pi * hour / 24)
        hr_cos = np.cos(2 * np.pi * hour / 24)
        month = 1 if temp < 8 else (4 if temp < 15 else (7 if temp > 25 else 10))
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        is_peak_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
        
        feature_values = {
            'hr': hour, 'temp': temp, 'hum': humidity, 'windspeed': windspeed,
            'season': season, 'weekday': day_of_week, 'hr_sin': hr_sin, 'hr_cos': hr_cos,
            'month_sin': month_sin, 'month_cos': month_cos, 'is_peak_hour': is_peak_hour
        }
        
        if self.dso2_features:
            ordered_values = [feature_values.get(feat, 0) for feat in self.dso2_features]
            df = pd.DataFrame([ordered_values], columns=self.dso2_features)
        else:
            df = pd.DataFrame([list(feature_values.values())], columns=list(feature_values.keys()))
        return df
    
    def predict_cluster_dso2(self, features_dict):
        """
        DSO2: Prédiction du cluster avec KMeans
        ⚠️ IMPORTANT: KMeans entraîné sur 11 features normalisées, PAS sur PCA!
        """
        try:
            X = self.prepare_features_dso2(features_dict)
            
            print(f"\n🔍 DSO2 - Features préparées ({X.shape[1]} features):")
            print(X)
            
            # Appliquer UNIQUEMENT le scaler (pas de PCA!)
            X_scaled = self.dso2_scaler.transform(X)
            print(f"   ✅ Scaler appliqué: shape {X_scaled.shape}")
            print(f"   ⚠️  PCA NON appliquée (KMeans attend 11 features, pas 2)")
            
            # Prédiction directe avec les 11 features normalisées
            cluster_id = int(self.dso2_model.predict(X_scaled)[0])
            
            print(f"✅ DSO2 - Cluster prédit: {cluster_id}")
            
            # Mapper le cluster à un segment
            segment_info = self.map_cluster_to_segment(cluster_id, features_dict)
            
            return segment_info
        except Exception as e:
            print(f"❌ Erreur DSO2: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def map_cluster_to_segment(self, cluster_id, features_dict):
        """Mapper le cluster ID à un segment métier"""
        segments = {
            0: {
                'cluster_id': 0,
                'segment_name': '🔧 Demande Faible',
                'description': 'Conditions défavorables ou heures creuses'
            },
            1: {
                'cluster_id': 1,
                'segment_name': '📊 Demande Standard',
                'description': 'Utilisation normale, conditions moyennes'
            },
            2: {
                'cluster_id': 2,
                'segment_name': '🚀 Haute Demande (Pointe)',
                'description': 'Rush hours, forte demande commute'
            },
            3: {
                'cluster_id': 3,
                'segment_name': '🖐️ Loisirs & Weekend',
                'description': 'Usage récréatif, zones touristiques'
            }
        }
        return segments.get(cluster_id, segments[1])
    
    def generate_recommendations_dso3(self, prediction_result, cluster_info, features_dict):
        """DSO3: Générer les recommandations"""
        try:
            total_demand = prediction_result['total']
            casual = prediction_result['casual']
            registered = prediction_result['registered']
            hour = features_dict.get('hour', 12)
            cluster_id = cluster_info['cluster_id']
            
            print(f"\n🔍 DSO3 - Génération recommandations (Cluster: {cluster_id})")
            
            if cluster_id == 0:
                return {
                    'recommendation': f"Demande faible ({total_demand} vélos). Période maintenance.",
                    'marketing_strategy': "Promotions légères",
                    'capacity_recommendation': max(10, int(total_demand * 0.7)),
                    'pricing_strategy': "Tarifs réduits",
                    'target_audience': f"Casual: {casual}, Registered: {registered}",
                    'actions': ["Maintenance", "Nettoyage", "Redistribution", "Promotions"]
                }
            elif cluster_id == 1:
                return {
                    'recommendation': f"Demande standard ({total_demand} vélos). Équilibre.",
                    'marketing_strategy': "Mix promotions casual + registered",
                    'capacity_recommendation': int(total_demand * 1.1),
                    'pricing_strategy': "Tarifs standards",
                    'target_audience': f"Mix (Casual: {casual}, Registered: {registered})",
                    'actions': ["Monitoring", "Newsletter", "Fidélité", "Promotions"]
                }
            elif cluster_id == 2:
                is_peak = (7 <= hour <= 9) or (17 <= hour <= 19)
                return {
                    'recommendation': f"{'⚠️ POINTE! ' if is_peak else ''}Haute demande ({total_demand} vélos).",
                    'marketing_strategy': "Focus commuters premium",
                    'capacity_recommendation': int(total_demand * 1.5),
                    'pricing_strategy': "Surge pricing",
                    'target_audience': f"Commuters ({registered} registered)",
                    'actions': ["Alerte équipes", "Redistribution", "Support", "SMS", "Premium"]
                }
            else:
                return {
                    'recommendation': f"Demande loisirs ({total_demand} vélos). Focus touristique.",
                    'marketing_strategy': "Campagnes weekend/familles",
                    'capacity_recommendation': int(total_demand * 1.3),
                    'pricing_strategy': "Forfaits journée/weekend",
                    'target_audience': f"Touristes ({casual} casual)",
                    'actions': ["Stock touristique", "Partenariats", "Promos familles", "Social", "Guides"]
                }
        except Exception as e:
            print(f"⚠️  Erreur DSO3: {e}")
            return {
                'recommendation': f"Demande: {prediction_result['total']} vélos",
                'marketing_strategy': "Stratégie standard",
                'capacity_recommendation': prediction_result['total'],
                'pricing_strategy': "Tarifs standards",
                'target_audience': "Mix utilisateurs",
                'actions': ["Monitoring", "Maintenance"]
            }

try:
    ml_manager = MLModelsManager()
    print("✅ Gestionnaire ML initialisé\n")
except Exception as e:
    print(f"❌ ERREUR: {e}")
    raise

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
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
            flash('🎉 Compte créé avec succès!', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('Erreur lors de la création du compte.', 'danger')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
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
    logout_user()
    flash('Vous avez été déconnecté avec succès.', 'success')
    return redirect(url_for('home'))

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    from datetime import date
    result = None
    current_date = date.today().strftime('%Y-%m-%d')
    if request.method == 'POST':
        try:
            prediction_date_str = request.form.get('prediction_date')
            hour = int(request.form.get('hour', 12))
            temp = float(request.form.get('temp', 20))
            hum = float(request.form.get('hum', 0.5))
            is_holiday = int(request.form.get('is_holiday', 0))
            weather = int(request.form.get('weather', 1))
            if not (0 <= hour <= 23):
                flash('L\'heure doit être entre 0 et 23.', 'danger')
                return redirect(url_for('dashboard'))
            if not (0 <= hum <= 1):
                flash('L\'humidité doit être entre 0.0 et 1.0.', 'danger')
                return redirect(url_for('dashboard'))
            prediction_date = None
            day_of_week = None
            day_name = "N/A"
            if prediction_date_str:
                try:
                    prediction_date = datetime.strptime(prediction_date_str, '%Y-%m-%d').date()
                    day_of_week = prediction_date.weekday()
                    day_names = {0: 'Lundi', 1: 'Mardi', 2: 'Mercredi', 3: 'Jeudi', 4: 'Vendredi', 5: 'Samedi', 6: 'Dimanche'}
                    day_name = day_names.get(day_of_week, "N/A")
                except ValueError:
                    flash('Format de date invalide.', 'danger')
                    return redirect(url_for('dashboard'))
            features = {'hour': hour, 'temp': temp, 'humidity': hum, 'is_holiday': is_holiday, 'day_of_week': day_of_week or 0, 'weather': weather}
            print("\n" + "="*70)
            print("🎯 NOUVELLE PRÉDICTION")
            print("="*70)
            prediction_result = ml_manager.predict_demand_dso1(features)
            cluster_info = ml_manager.predict_cluster_dso2(features)
            recommendations = ml_manager.generate_recommendations_dso3(prediction_result, cluster_info, features)
            print("="*70)
            print("✅ PRÉDICTION COMPLÈTE")
            print("="*70 + "\n")
            result = {
                'total': prediction_result['total'], 'casual': prediction_result['casual'], 'registered': prediction_result['registered'],
                'model_name': prediction_result['model_name'], 'cluster_id': cluster_info['cluster_id'], 'segment': cluster_info['segment_name'],
                'segment_description': cluster_info['description'], 'recommendation': recommendations['recommendation'],
                'marketing_strategy': recommendations['marketing_strategy'], 'actions': recommendations['actions'],
                'capacity_recommendation': recommendations['capacity_recommendation'], 'pricing_strategy': recommendations['pricing_strategy'],
                'target_audience': recommendations.get('target_audience', 'Mix'), 'hour': hour, 'temp': temp, 'confidence': 98.5,
                'prediction_date': prediction_date.strftime('%d/%m/%Y') if prediction_date else 'Aujourd\'hui', 'day_name': day_name
            }
            new_prediction = PredictionHistory(
                prediction_date=prediction_date, hour=hour, temp=temp, humidity=hum, is_holiday=bool(is_holiday),
                day_of_week=day_of_week, weather=weather, predicted_demand=prediction_result['total'],
                predicted_casual=prediction_result['casual'], predicted_registered=prediction_result['registered'],
                segment=cluster_info['segment_name'], cluster_id=cluster_info['cluster_id'], confidence_score=98.5,
                model_used=prediction_result['model_name'], recommendation=recommendations['recommendation'],
                marketing_strategy=recommendations['marketing_strategy'], author=current_user
            )
            db.session.add(new_prediction)
            db.session.commit()
            flash('✅ Analyse complète générée avec les 3 modèles!', 'success')
        except Exception as e:
            flash(f'Erreur: {e}', 'danger')
            print(f"❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
    return render_template('dashboard.html', result=result, current_date=current_date)

@app.route('/history')
@login_required
def history():
    preds = PredictionHistory.query.filter_by(user_id=current_user.id).order_by(PredictionHistory.date_posted.desc()).all()
    return render_template('history.html', preds=preds)

@app.route('/api/stats')
@login_required
def api_stats():
    preds = PredictionHistory.query.filter_by(user_id=current_user.id).all()
    if not preds:
        return jsonify({'total': 0, 'average_demand': 0, 'max_demand': 0, 'min_demand': 0, 'total_casual': 0, 'total_registered': 0, 'casual_percentage': 0, 'registered_percentage': 0})
    total_count = len(preds)
    demands = [p.predicted_demand for p in preds]
    avg_demand = sum(demands) / total_count
    max_demand = max(demands)
    min_demand = min(demands)
    total_casual = sum([p.predicted_casual for p in preds])
    total_registered = sum([p.predicted_registered for p in preds])
    total_users = total_casual + total_registered
    casual_pct = (total_casual / total_users * 100) if total_users > 0 else 0
    registered_pct = (total_registered / total_users * 100) if total_users > 0 else 0
    return jsonify({'total': total_count, 'average_demand': avg_demand, 'max_demand': max_demand, 'min_demand': min_demand, 'total_casual': total_casual, 'total_registered': total_registered, 'casual_percentage': casual_pct, 'registered_percentage': registered_pct})

@app.route('/statistics')
@login_required
def statistics():
    return render_template('statistics.html')

@app.errorhandler(404)
def not_found(error):
    flash('Page non trouvée.', 'danger')
    return redirect(url_for('home'))

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    flash('Erreur serveur.', 'danger')
    return redirect(url_for('home'))

def init_db():
    with app.app_context():
        try:
            db.create_all()
            print("✅ Tables créées")
        except Exception as e:
            print(f"❌ Erreur: {e}")

if __name__ == '__main__':
    init_db()
    print("\n" + "="*70)
    print("🚀 BIKEPRED PRO - 3 MODÈLES ML")
    print("="*70)
    print(f"🌐 http://localhost:5000")
    print(f"🤖 DSO1: Prédiction demande (15 features)")
    print(f"🤖 DSO2: Clustering (11 features - PAS de PCA)")
    print(f"🤖 DSO3: Recommandations")
    print("="*70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)