import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np

def train_and_save_model():
    """
    Entraîne un modèle simple sur des données factices et le sauvegarde.
    """
    print("Début de l'entraînement...")
    # Données factices : [taille, poids] -> 0 (chat), 1 (chien)
    X = np.array([[10, 2], [12, 3], [40, 10], [45, 12]])
    y = np.array([0, 0, 1, 1])

    # Création et entraînement du modèle
    model = LogisticRegression()
    model.fit(X, y)

    # Sauvegarde du modèle dans un fichier
    model_filename = 'simple_model.joblib'
    joblib.dump(model, model_filename)
    print(f"Modèle entraîné et sauvegardé sous le nom : {model_filename}")
    return model_filename

def predict(model_path, data):
    """
    Charge un modèle et fait une prédiction.
    """
    try:
        # Chargement du modèle
        model = joblib.load(model_path)
        print(f"Modèle chargé depuis {model_path}")

        # Prédiction
        prediction = model.predict(data)
        print(f"Prédiction pour les données {data} : {prediction}")
        return prediction
    except FileNotFoundError:
        print(f"Erreur : Le fichier du modèle '{model_path}' n'a pas  été trouvé.")
        return None

# --- Point d'entrée du script ---
if __name__ == '__main__':
    # 1. Entraîner et sauvegarder le modèle
    saved_model_path = train_and_save_model()

    # 2. Utiliser le modèle pour faire une prédiction
    if saved_model_path:
        # Données de test (par exemple, un animal de [50, 15])
        new_data = np.array([[50, 15]])
        predict(saved_model_path, new_data)