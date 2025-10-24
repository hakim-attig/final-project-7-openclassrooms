import unittest
import joblib
import pandas as pd
import numpy as np

class TestPreprocessing(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Charger l'imputer"""
        cls.imputer = joblib.load("models/imputer.pkl")
        cls.features = joblib.load("models/feature_columns.pkl")
    
    def test_imputer_loaded(self):
        """Test: Imputer chargé"""
        self.assertIsNotNone(self.imputer)
        
    def test_imputation(self):
        """Test: Imputation des NaN"""
        X_test = pd.DataFrame(np.random.randn(5, 254), columns=self.features)
        X_test.iloc[0, 0] = np.nan
        
        X_imputed = self.imputer.transform(X_test)
        self.assertFalse(np.isnan(X_imputed).any())

if __name__ == "__main__":
    unittest.main()