import unittest
from src.main import HuggingFaceModel, predict_with_model

class TestHuggingFaceModel(unittest.TestCase):
    def setUp(self):
        self.model = HuggingFaceModel()

    def test_predict_output(self):
        text = "un arbol corto las cuerdas"
        predictions = self.model.predict(text)
        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0)
        for prob in predictions[0]:
            self.assertGreaterEqual(prob, 0)
            self.assertLessEqual(prob, 1)
    
class TestGradioInterface(unittest.TestCase):
    def test_predict_with_model(self):
        input_text = "no hay energia en el sector"
        predictions = predict_with_model(input_text)
        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0)
        for prob in predictions[0]:
            self.assertGreaterEqual(prob, 0)
            self.assertLessEqual(prob, 1)   

if __name__ == "__main__":
    unittest.main()