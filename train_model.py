import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def accuracy_score_test():
    data = {
        'taille': [170, 165, 180, 175, 160, 185, 170, 155],
        'poids': [65, 60, 80, 75, 55, 90, 68, 50],
        'sportif': [1, 0, 1, 1, 0, 1, 0, 0]
    }
  
    df = pd.DataFrame(data)

    # Séparation des variables
    X = df[['taille', 'poids']]
    y = df['sportif']

    # Division en jeu d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Entraînement du modèle
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Prédiction
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Précision du modèle : {accuracy*100:.2f}%")

    return accuracy

if __name__ == "__main__":
    accuracy_score_test()


