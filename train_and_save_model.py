"""
Script pour entraîner et sauvegarder le modèle Random Forest
À exécuter UNE FOIS pour générer les fichiers .pkl
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
import os

print("🚀 Démarrage de l'entraînement des modèles ML...")

# =============================================================================
# 1. CHARGEMENT DES DONNÉES
# =============================================================================

df = pd.read_csv('bike_sharing_final_with_clusters.csv')
print(f"✅ Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")

# =============================================================================
# 2. PRÉPARATION DES FEATURES POUR LA RÉGRESSION (DSO1)
# =============================================================================

# Features pour prédire la demande (cnt)
regression_features = ['hr', 'temp', 'hum', 'is_holiday', 'yr', 'season', 
                       'is_peak_hour', 'is_night', 'bad_weather', 
                       'temp_windspeed_interaction', 'hr_sin', 'hr_cos']

# Sélectionner uniquement les features disponibles
available_features = [f for f in regression_features if f in df.columns]
print(f"📊 Features disponibles pour régression: {len(available_features)}")

# Si certaines colonnes manquent, créer des features basiques
if 'is_holiday' not in df.columns:
    df['is_holiday'] = 0
if 'yr' not in df.columns:
    df['yr'] = 0
if 'season' not in df.columns:
    df['season'] = 1

# Préparer X et y pour la régression
X_reg = df[available_features].fillna(0)
y_reg = df['cnt']

# Split train/test
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print(f"✅ Train: {len(X_train_reg)}, Test: {len(X_test_reg)}")

# =============================================================================
# 3. ENTRAÎNEMENT DU MODÈLE DE RÉGRESSION (Random Forest)
# =============================================================================

print("\n🌲 Entraînement du Random Forest Regressor...")

rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

rf_regressor.fit(X_train_reg, y_train_reg)

# Évaluation
train_score = rf_regressor.score(X_train_reg, y_train_reg)
test_score = rf_regressor.score(X_test_reg, y_test_reg)

print(f"✅ R² Train: {train_score:.4f}")
print(f"✅ R² Test: {test_score:.4f}")

# Prédictions de test
y_pred_test = rf_regressor.predict(X_test_reg)
rmse = np.sqrt(np.mean((y_test_reg - y_pred_test)**2))
mae = np.mean(np.abs(y_test_reg - y_pred_test))

print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")

# =============================================================================
# 4. ENTRAÎNEMENT DU MODÈLE DE CLUSTERING (KMeans)
# =============================================================================

print("\n🎯 Entraînement du KMeans Clustering...")

# Features pour clustering (si pas déjà dans le CSV)
if 'cluster' not in df.columns:
    clustering_features = ['hr', 'temp', 'hum', 'cnt']
    X_cluster = df[clustering_features].fillna(0)
    
    # Standardisation
    scaler_cluster = StandardScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
    
    # KMeans
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_cluster_scaled)
    
    # Sauvegarder
    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    joblib.dump(scaler_cluster, 'models/scaler_cluster.pkl')
    print("✅ KMeans et Scaler sauvegardés")
else:
    print("✅ Clusters déjà présents dans le CSV")

# =============================================================================
# 5. ENTRAÎNEMENT DU MODÈLE DE CLASSIFICATION (DSO3)
# =============================================================================

print("\n🎯 Entraînement du Random Forest Classifier...")

# Créer une variable cible pour la classification
# Par exemple: classifier si c'est une période "haute demande" ou non
df['high_demand'] = (df['cnt'] > df['cnt'].quantile(0.75)).astype(int)

X_class = df[available_features].fillna(0)
y_class = df['high_demand']

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)

rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf_classifier.fit(X_train_class, y_train_class)

accuracy = rf_classifier.score(X_test_class, y_test_class)
print(f"✅ Accuracy: {accuracy:.4f}")

# =============================================================================
# 6. SAUVEGARDE DES MODÈLES ET MÉTADONNÉES
# =============================================================================

print("\n💾 Sauvegarde des modèles...")

# Créer le dossier models s'il n'existe pas
os.makedirs('models', exist_ok=True)

# Sauvegarder les modèles
joblib.dump(rf_regressor, 'models/rf_regressor.pkl')
joblib.dump(rf_classifier, 'models/rf_classifier.pkl')

# Sauvegarder les métadonnées
metadata = {
    'regression_features': available_features,
    'regression_r2_test': float(test_score),
    'regression_rmse': float(rmse),
    'regression_mae': float(mae),
    'classification_accuracy': float(accuracy),
    'n_samples': len(df),
    'date_trained': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

joblib.dump(metadata, 'models/model_metadata.pkl')

print("✅ Tous les modèles sauvegardés:")
print("   - models/rf_regressor.pkl")
print("   - models/rf_classifier.pkl")
print("   - models/model_metadata.pkl")

# =============================================================================
# 7. TEST DU MODÈLE SAUVEGARDÉ
# =============================================================================

print("\n🧪 Test de chargement du modèle...")

loaded_model = joblib.load('models/rf_regressor.pkl')
loaded_metadata = joblib.load('models/model_metadata.pkl')

# Prédiction test
test_input = X_test_reg.iloc[0:1]
test_pred = loaded_model.predict(test_input)

print(f"✅ Modèle chargé avec succès!")
print(f"   Test prédiction: {test_pred[0]:.0f} vélos")
print(f"   Valeur réelle: {y_test_reg.iloc[0]:.0f} vélos")
print(f"   Erreur: {abs(test_pred[0] - y_test_reg.iloc[0]):.0f} vélos")

print("\n" + "="*50)
print("🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
print("="*50)
print(f"📊 Métriques finales:")
print(f"   - R² Test: {metadata['regression_r2_test']:.4f}")
print(f"   - RMSE: {metadata['regression_rmse']:.2f}")
print(f"   - MAE: {metadata['regression_mae']:.2f}")
print(f"   - Accuracy Classification: {metadata['classification_accuracy']:.4f}")
print("="*50)