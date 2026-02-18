import pickle

print("="*70)
print("🔍 VÉRIFICATION DE LA STRUCTURE DES MODÈLES")
print("="*70)

# DSO1
print("\n📦 DSO1_prediction_demande.pkl:")
with open('DSO1_prediction_demande.pkl', 'rb') as f:
    dso1 = pickle.load(f)
    print(f"Type: {type(dso1)}")
    if isinstance(dso1, dict):
        print(f"Clés: {list(dso1.keys())}")
        for key, value in dso1.items():
            print(f"  - {key}: {type(value).__name__}")
    else:
        print(f"C'est directement un: {type(dso1).__name__}")

# DSO2
print("\n📦 DSO2_segmentation_clusters.pkl:")
with open('DSO2_segmentation_clusters.pkl', 'rb') as f:
    dso2 = pickle.load(f)
    print(f"Type: {type(dso2)}")
    if isinstance(dso2, dict):
        print(f"Clés: {list(dso2.keys())}")
        for key, value in dso2.items():
            print(f"  - {key}: {type(value).__name__}")
    else:
        print(f"C'est directement un: {type(dso2).__name__}")

# DSO3
print("\n📦 DSO3_recommandations_marketing.pkl:")
with open('DSO3_recommandations_marketing.pkl', 'rb') as f:
    dso3 = pickle.load(f)
    print(f"Type: {type(dso3)}")
    if isinstance(dso3, dict):
        print(f"Clés: {list(dso3.keys())}")
        for key, value in dso3.items():
            print(f"  - {key}: {type(value).__name__}")
    else:
        print(f"C'est directement un: {type(dso3).__name__}")

print("\n" + "="*70)