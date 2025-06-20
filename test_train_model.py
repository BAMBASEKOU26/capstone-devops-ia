import pytest
import train_model

def test_accuracy():
    # On exécute le script principal et on récupère la précision
    accuracy = train_model.accuracy_score_test()
    assert accuracy >= 0 and accuracy <= 1, "La précision doit être entre 0 et 1"
    assert accuracy >= 0.5, "La précision doit être au moins de 50%"
