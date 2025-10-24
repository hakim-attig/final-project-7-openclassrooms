import unittest
import numpy as np
from sklearn.metrics import confusion_matrix

class TestBusinessLogic(unittest.TestCase):
    
    def test_cost_calculation(self):
        """Test: Calcul coût métier 10*FN + FP"""
        fn, fp = 100, 500
        cost = 10 * fn + fp
        self.assertEqual(cost, 1500)
        
    def test_threshold_logic(self):
        """Test: Application du seuil"""
        probas = np.array([0.05, 0.08, 0.09, 0.12])
        threshold = 0.09
        predictions = (probas >= threshold).astype(int)
        expected = np.array([0, 0, 0, 1])
        np.testing.assert_array_equal(predictions, expected)
        
    def test_confusion_matrix_cost(self):
        """Test: Coût depuis matrice de confusion"""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 0])
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = 10 * fn + fp
        
        # FN=2 (indices 2,5), FP=1 (indice 1)
        self.assertEqual(fn, 2)
        self.assertEqual(fp, 1)
        self.assertEqual(cost, 21)

if __name__ == "__main__":
    unittest.main()