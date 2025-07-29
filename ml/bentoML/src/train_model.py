
import os
import pandas as pd
from pycaret.classification import (
    setup, compare_models, create_model,
    tune_model, evaluate_model, plot_model,
    interpret_model, finalize_model, predict_model, save_model, get_config
)
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import bentoml

# 1. Charger les données
def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")
    return pd.read_excel(filepath)

# 2. Nettoyer et transformer les données
def preprocess_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    id_cols = [col for col in df.columns if 'id' in col.lower()]
    df = df.drop(columns=id_cols, errors='ignore')
    # Colonnes probables (à ajuster après vérification)
    possible_cols = [
        'Business Travel', 'CF_age band', 'Job Role', 'Marital Status',
        'Monthly Income', 'Years At Company', 'Department', 'Education'
    ]
    # Vérifier les colonnes disponibles
    selected_cols = [col for col in possible_cols if col in df.columns]
    if not selected_cols:
        raise ValueError(f"Aucune colonne valide trouvée parmi : {possible_cols}")
    print(f"Colonnes sélectionnées : {selected_cols}")
    if 'Attrition' in df.columns:
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        X = df[selected_cols]
        y = df['Attrition']
        return X, y
    raise ValueError("Colonne 'Attrition' non trouvée.")

# 3. Entraîner le modèle
def train_model(X, y, target='y'):
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("✅ Initialisation de PyCaret...")
    train_data = pd.concat([X_train, y_train.rename(target)], axis=1)
    setup_config = setup(
        data=train_data,
        target=target,
        session_id=123,
        verbose=False,
        html=False,
        log_experiment=False,
        profile=False,
        preprocess=True,
        fix_imbalance=False,
        fold=10
    )

    print("✅ Comparaison des modèles...")
    best_model = compare_models()
    print(f"Meilleur modèle : {best_model}")

    print("✅ Création d'un modèle Logistic Regression...")
    lr_model = create_model('lr')

    print("✅ Optimisation du modèle...")
    tuned_model = tune_model(lr_model)

    print("📊 Évaluation interactive du modèle...")
    try:
        evaluate_model(tuned_model)
        print("✅ Évaluation interactive ouverte.")
    except Exception as e:
        print(f"⚠️ Erreur lors de l'évaluation interactive : {str(e)}")
        print("💡 Suggestion : Utilisez plot_model pour des visualisations spécifiques.")

    print("📊 Sauvegarde des visualisations...")
    plot_model(tuned_model, plot='confusion_matrix', save=True)
    print("✅ Matrice de confusion sauvegardée.")
    
    try:
        plot_model(tuned_model, plot='auc', save=True)
        print("✅ Courbe AUC sauvegardée.")
    except Exception as e:
        print(f"⚠️ Erreur lors de la génération de la courbe AUC : {str(e)}")

    plot_model(tuned_model, plot='feature', save=True)
    print("✅ Importance des caractéristiques sauvegardée.")

    print("🔍 Interprétation SHAP...")
    try:
        interpret_model(tuned_model)
        print("✅ Interprétation SHAP réussie.")
    except Exception as e:
        print(f"⚠️ SHAP non supporté pour ce modèle : {str(e)}")

    print("🔍 Génération des prédictions sur l'ensemble de test...")
    test_data = pd.concat([X_test, y_test.rename(target)], axis=1)
    predictions = predict_model(tuned_model, data=test_data)
    print("Exemple de prédictions :")
    print(predictions.head())

    print("✅ Finalisation du modèle...")
    final_model = finalize_model(tuned_model)

    print("💾 Sauvegarde du modèle (PyCaret)...")
    save_model(final_model, 'best_model')
    print("✅ Modèle PyCaret sauvegardé.")

    print("💾 Sauvegarde du modèle pour BentoML...")
    try:
        bentoml.pycaret.save_model("employee_attrition_model", final_model)
        print("✅ Modèle sauvegardé pour BentoML.")
    except Exception as e:
        print(f"⚠️ Erreur lors de la sauvegarde BentoML : {str(e)}")
        print("⚠️ Utilisation du fallback avec bentoml.save_model...")
        bentoml.save_model("employee_attrition_model", final_model)
        print("✅ Modèle sauvegardé pour BentoML avec bentoml.save_model.")

    return final_model

# 4. Script principal
if __name__ == "__main__":
    excel_path = os.path.abspath("../data/employees.xlsx")
    try:
        df = load_data(excel_path)
        print(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        X, y = preprocess_data(df)
        print("Distribution des classes dans 'Attrition' :")
        print(y.value_counts())
        print("✅ Données prétraitées.")
        trained_model = train_model(X, y)
    except FileNotFoundError as e:
        print(f"❌ Erreur : {e}")
    except Exception as e:
        print(f"❌ Une erreur s'est produite : {str(e)}")