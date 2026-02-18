# check_dso2_features.py
import pickle

print("🔍 Vérification DSO2 en détail")
print("="*70)

with open('DSO2_segmentation_clusters.pkl', 'rb') as f:
    dso2 = pickle.load(f)

print(f"Features dans le fichier: {dso2['features']}")
print(f"Nombre de features: {len(dso2['features'])}")

if dso2['pca']:
    print(f"\nPCA présent:")
    print(f"  - n_components: {dso2['pca'].n_components_}")
    print(f"  - n_features_in: {dso2['pca'].n_features_in_}")

print(f"\nKMeans:")
print(f"  - n_clusters: {dso2['model'].n_clusters}")
print(f"  - n_features_in: {dso2['model'].n_features_in_}")

print(f"\nScaler:")
if dso2['scaler']:
    print(f"  - n_features_in: {dso2['scaler'].n_features_in_}")
    print(f"  - feature_names: {dso2['scaler'].feature_names_in_ if hasattr(dso2['scaler'], 'feature_names_in_') else 'N/A'}")