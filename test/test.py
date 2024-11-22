import unittest
from src.main import HuggingFaceModel, predict_with_model

class TestHuggingFaceModel(unittest.TestCase):
    def setUp(self):
        self.model = HuggingFaceModel()

    def test_predict_output(self):
        # Probar con una entrada válida
        text = "un arbol corto las cuerdas"
        predictions = self.model.predict(text)
        self.assertIsInstance(predictions, str)  # Verificar que la salida es un string
        self.assertIn(predictions, ["ejecutar", "cancelar"])  # Validar que es una etiqueta válida
    
    def test_invalid_input(self):
        # Probar con entrada inválida
        invalid_text = "12"  # Menos de 3 caracteres o puramente numérico
        result = self.model.predict(invalid_text)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Entrada inválida. Por favor, ingrese un texto coherente.")

class TestGradioInterface(unittest.TestCase):
    def test_predict_with_model(self):
        # Probar con texto para la interfaz Gradio
        input_text = "no hay energia en el sector"
        predictions = predict_with_model(input_text)
        self.assertIsInstance(predictions, str)  # Verificar que la salida es un string
        self.assertIn(predictions, ["ejecutar", "cancelar"])  # Validar que es una etiqueta válida

if __name__ == "__main__":
    unittest.main()