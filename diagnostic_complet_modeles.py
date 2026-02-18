"""
Script de diagnostic complet pour vérifier les 3 modèles .pkl
Exécutez ce script AVANT de lancer l'application
"""

import pickle
import os
import pandas as pd
import numpy as np

def check_file_exists(filepath):
    """Vérifier si le fichier existe"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / 1024  # En KB
        print(f"   ✅ Fichier trouvé: {filepath} ({size:.2f} KB)")
        return True
    else:
        print(f"   ❌ Fichier manquant: {filepath}")
        return False


def analyze_dso1():
    """Analyser DSO1_prediction_demande.pkl"""
    print("\n" + "="*70)
    print("🔍 ANALYSE DSO1 - PRÉDICTION DE DEMANDE")
    print("="*70)
    
    filepath = 'DSO1_prediction_demande.pkl'
    
    if not check_file_exists(filepath):
        return False
    
    try:
        with open(filepath, 'rb') as f:
            dso1 = pickle.load(f)
        
        print(f"\n📦 Type du contenu: {type(dso1)}")
        
        if isinstance(dso1, dict):
            print(f"   Clés disponibles: {list(dso1.keys())}")
            
            # Analyser le modèle
            if 'model' in dso1:
                model = dso1['model']
                print(f"\n🤖 Modèle:")
                print(f"   Type: {type(model).__name__}")
                print(f"   Nombre de features attendues: {model.n_features_in_}")
                
                if hasattr(model, 'feature_names_in_'):
                    print(f"   Noms des features: {list(model.feature_names_in_)}")
            
            # Analyser le scaler
            if 'scaler' in dso1 and dso1['scaler'] is not None:
                scaler = dso1['scaler']
                print(f"\n📏 Scaler:")
                print(f"   Type: {type(scaler).__name__}")
                print(f"   Nombre de features: {scaler.n_features_in_}")
                
                if hasattr(scaler, 'feature_names_in_'):
                    print(f"   Noms des features: {list(scaler.feature_names_in_)}")
            
            # Analyser la liste des features
            if 'features' in dso1:
                features = dso1['features']
                print(f"\n📋 Liste des features ({len(features)} features):")
                for i, feat in enumerate(features, 1):
                    print(f"   {i}. {feat}")
        
        else:
            print(f"   ⚠️  Format inattendu: modèle directement dans le fichier")
            print(f"   Type: {type(dso1).__name__}")
            if hasattr(dso1, 'n_features_in_'):
                print(f"   Features attendues: {dso1.n_features_in_}")
        
        # Test de prédiction
        print(f"\n🧪 Test de prédiction basique...")
        
        # Préparer des données de test (adapter selon les features trouvées)
        if isinstance(dso1, dict) and 'features' in dso1:
            test_data = {}
            for feat in dso1['features']:
                if feat in ['hr', 'hour']:
                    test_data[feat] = 12
                elif feat == 'temp':
                    test_data[feat] = 20
                elif feat in ['hum', 'humidity']:
                    test_data[feat] = 0.5
                elif feat in ['weathersit', 'weather']:
                    test_data[feat] = 1
                elif feat in ['holiday', 'is_holiday']:
                    test_data[feat] = 0
                elif feat in ['weekday', 'day_of_week']:
                    test_data[feat] = 2
                elif feat == 'workingday':
                    test_data[feat] = 1
                else:
                    test_data[feat] = 0
            
            X_test = pd.DataFrame([test_data])
            print(f"   Features de test:")
            print(X_test)
            
            # Appliquer scaler si présent
            model = dso1['model'] if isinstance(dso1, dict) else dso1
            scaler = dso1.get('scaler') if isinstance(dso1, dict) else None
            
            if scaler:
                X_test_scaled = scaler.transform(X_test)
                print(f"   ✅ Scaler appliqué")
            else:
                X_test_scaled = X_test.values
            
            # Prédiction
            prediction = model.predict(X_test_scaled)[0]
            print(f"   ✅ Prédiction réussie: {int(prediction)} vélos")
        
        print("\n✅ DSO1 analysé avec succès!")
        return True
    
    except Exception as e:
        print(f"\n❌ Erreur lors de l'analyse DSO1: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_dso2():
    """Analyser DSO2_segmentation_clusters.pkl"""
    print("\n" + "="*70)
    print("🔍 ANALYSE DSO2 - SEGMENTATION/CLUSTERING")
    print("="*70)
    
    filepath = 'DSO2_segmentation_clusters.pkl'
    
    if not check_file_exists(filepath):
        return False
    
    try:
        with open(filepath, 'rb') as f:
            dso2 = pickle.load(f)
        
        print(f"\n📦 Type du contenu: {type(dso2)}")
        
        if isinstance(dso2, dict):
            print(f"   Clés disponibles: {list(dso2.keys())}")
            
            # Analyser le modèle de clustering
            if 'model' in dso2:
                model = dso2['model']
                print(f"\n🤖 Modèle de clustering:")
                print(f"   Type: {type(model).__name__}")
                print(f"   Nombre de features attendues: {model.n_features_in_}")
                
                if hasattr(model, 'n_clusters'):
                    print(f"   Nombre de clusters: {model.n_clusters}")
            
            # Analyser le scaler
            if 'scaler' in dso2 and dso2['scaler'] is not None:
                scaler = dso2['scaler']
                print(f"\n📏 Scaler:")
                print(f"   Type: {type(scaler).__name__}")
                print(f"   Nombre de features: {scaler.n_features_in_}")
            
            # Analyser PCA
            if 'pca' in dso2 and dso2['pca'] is not None:
                pca = dso2['pca']
                print(f"\n🔬 PCA (Réduction de dimensionnalité):")
                print(f"   Type: {type(pca).__name__}")
                print(f"   Features d'entrée: {pca.n_features_in_}")
                print(f"   Composantes principales: {pca.n_components_}")
                print(f"   Variance expliquée: {pca.explained_variance_ratio_.sum():.2%}")
            
            # Analyser la liste des features
            if 'features' in dso2:
                features = dso2['features']
                print(f"\n📋 Liste des features ({len(features)} features):")
                for i, feat in enumerate(features, 1):
                    print(f"   {i}. {feat}")
        
        # Test de prédiction
        print(f"\n🧪 Test de clustering basique...")
        
        if isinstance(dso2, dict) and 'features' in dso2:
            # Préparer des données de test avec les 11 features
            test_data = {}
            for feat in dso2['features']:
                if feat in ['hr', 'hour']:
                    test_data[feat] = 18
                elif feat == 'temp':
                    test_data[feat] = 22
                elif feat in ['hum', 'humidity']:
                    test_data[feat] = 0.6
                elif feat == 'windspeed':
                    test_data[feat] = 0.19
                elif feat == 'season':
                    test_data[feat] = 2
                elif feat in ['weekday', 'day_of_week']:
                    test_data[feat] = 2
                elif feat == 'hr_sin':
                    test_data[feat] = np.sin(2 * np.pi * 18 / 24)
                elif feat == 'hr_cos':
                    test_data[feat] = np.cos(2 * np.pi * 18 / 24)
                elif feat == 'month_sin':
                    test_data[feat] = np.sin(2 * np.pi * 6 / 12)
                elif feat == 'month_cos':
                    test_data[feat] = np.cos(2 * np.pi * 6 / 12)
                elif feat == 'is_peak_hour':
                    test_data[feat] = 1
                else:
                    test_data[feat] = 0
            
            X_test = pd.DataFrame([test_data])
            print(f"   Features de test:")
            print(X_test)
            
            # Pipeline de transformation
            model = dso2['model'] if isinstance(dso2, dict) else dso2
            scaler = dso2.get('scaler') if isinstance(dso2, dict) else None
            pca = dso2.get('pca') if isinstance(dso2, dict) else None
            
            # 1. Scaler
            if scaler:
                X_test_scaled = scaler.transform(X_test)
                print(f"   ✅ Scaler appliqué: shape {X_test_scaled.shape}")
            else:
                X_test_scaled = X_test.values
            
            # 2. PCA (si présent)
            if pca:
                X_test_transformed = pca.transform(X_test_scaled)
                print(f"   ✅ PCA appliquée: {X_test_scaled.shape} → {X_test_transformed.shape}")
            else:
                X_test_transformed = X_test_scaled
                print(f"   ℹ️  Pas de PCA, utilisation directe des features normalisées")
            
            # 3. Prédiction du cluster
            cluster = model.predict(X_test_transformed)[0]
            print(f"   ✅ Cluster prédit: {cluster}")
        
        print("\n✅ DSO2 analysé avec succès!")
        return True
    
    except Exception as e:
        print(f"\n❌ Erreur lors de l'analyse DSO2: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_dso3():
    """Analyser DSO3_recommandations_marketing.pkl"""
    print("\n" + "="*70)
    print("🔍 ANALYSE DSO3 - RECOMMANDATIONS MARKETING")
    print("="*70)
    
    filepath = 'DSO3_recommandations_marketing.pkl'
    
    if not check_file_exists(filepath):
        return False
    
    try:
        with open(filepath, 'rb') as f:
            dso3 = pickle.load(f)
        
        print(f"\n📦 Type du contenu: {type(dso3)}")
        
        if isinstance(dso3, dict):
            print(f"   Clés disponibles: {list(dso3.keys())}")
            
            # Analyser le contenu de chaque clé
            for key, value in dso3.items():
                print(f"\n📊 Clé '{key}':")
                print(f"   Type: {type(value)}")
                
                if isinstance(value, (list, tuple)):
                    print(f"   Nombre d'éléments: {len(value)}")
                    if len(value) > 0:
                        print(f"   Premiers éléments:")
                        for i, item in enumerate(value[:5], 1):
                            print(f"      {i}. {item}")
                elif isinstance(value, dict):
                    print(f"   Sous-clés: {list(value.keys())}")
                elif isinstance(value, pd.DataFrame):
                    print(f"   Shape: {value.shape}")
                    print(f"   Colonnes: {list(value.columns)}")
                else:
                    print(f"   Valeur: {value}")
        
        else:
            print(f"   ⚠️  Format inattendu: {type(dso3)}")
        
        print("\n✅ DSO3 analysé avec succès!")
        return True
    
    except Exception as e:
        print(f"\n❌ Erreur lors de l'analyse DSO3: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Fonction principale de diagnostic"""
    print("\n" + "="*70)
    print("🏥 DIAGNOSTIC COMPLET DES MODÈLES ML")
    print("="*70)
    
    results = {
        'DSO1': analyze_dso1(),
        'DSO2': analyze_dso2(),
        'DSO3': analyze_dso3()
    }
    
    # Résumé final
    print("\n" + "="*70)
    print("📋 RÉSUMÉ DU DIAGNOSTIC")
    print("="*70)
    
    for model, success in results.items():
        status = "✅ OK" if success else "❌ ÉCHEC"
        print(f"   {model}: {status}")
    
    all_ok = all(results.values())
    
    if all_ok:
        print("\n✅ TOUS LES MODÈLES SONT OPÉRATIONNELS!")
        print("   Vous pouvez lancer l'application Flask.")
    else:
        print("\n❌ CERTAINS MODÈLES ONT DES PROBLÈMES!")
        print("   Vérifiez les erreurs ci-dessus avant de lancer l'application.")
    
    print("="*70 + "\n")
    
    return all_ok


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)