import unittest
import joblib
import numpy as np
import pandas as pd

class TestChampionModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Charger le modèle une fois pour tous les tests"""
        cls.model = joblib.load("models/champion_model.pkl")
        cls.threshold = joblib.load("models/champion_threshold.pkl")
        cls.features = joblib.load("models/feature_columns.pkl")
        cls.metadata = joblib.load("models/model_metadata.pkl")
    
    def test_model_loaded(self):
        """Test: Le modèle est bien chargé"""
        self.assertIsNotNone(self.model)
        
    def test_threshold_range(self):
        """Test: Le seuil est dans [0, 1]"""
        self.assertGreaterEqual(self.threshold, 0)
        self.assertLessEqual(self.threshold, 1)
        
    def test_features_count(self):
        """Test: Nombre de features cohérent"""
        self.assertEqual(len(self.features), 254)
        
    def test_prediction_shape(self):
        """Test: Shape des prédictions"""
        X_test = pd.DataFrame(np.random.randn(10, 254), columns=self.features)
        pred = self.model.predict_proba(X_test)
        self.assertEqual(pred.shape, (10, 2))
        
    def test_prediction_range(self):
        """Test: Probabilités dans [0, 1]"""
        X_test = pd.DataFrame(np.random.randn(5, 254), columns=self.features)
        pred = self.model.predict_proba(X_test)
        self.assertTrue(np.all(pred >= 0))
        self.assertTrue(np.all(pred <= 1))

if __name__ == "__main__":
    unittest.main()