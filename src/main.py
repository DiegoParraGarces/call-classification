import re
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class HuggingFaceModel:
    def __init__(self, model_name="jcortizba/modelo18"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        # Mapeo de etiquetas
        self.label_mapping = {0: "ejecutar", 1: "cancelar"}

    def predict(self, text):
        # Validar entrada antes de predecir
        if not self.validate_input(text):
            return "Entrada inválida. Por favor, ingrese un texto coherente."
        
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        predicted_label = self.label_mapping[predicted_index]
        return predicted_label

    def validate_input(self, text):
        """Valida que el texto sea coherente y legible."""
        # Rechazar entradas vacías o con menos de 3 caracteres
        if len(text.strip()) < 3:
            return False
        # Permitir solo letras, números, espacios y signos de puntuación básicos
        if not re.match(r"^[a-zA-Z0-9ñÑ\s.,!?]+$", text):
        #if not re.match(r"^[a-zA-Z0-9áéíóúñÁÉÍÓÚÑüÜ\s.,!?]+$", text):
            return False
        # Verificar que no sea puramente numérico
        if text.strip().isdigit():
            return False
        return True

# Función para la interfaz
def predict_with_model(input_text):
    model = HuggingFaceModel()
    prediction = model.predict(input_text)
    return prediction

# Definir interfaz en Gradio
def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Modelo Hugging Face con Validación de Entrada")
        input_text = gr.Textbox(label="Ingrese texto para predecir")
        output_label = gr.Textbox(label="Etiqueta de predicción o mensaje de error")
        predict_button = gr.Button("Predecir")
        
        predict_button.click(
            fn=predict_with_model,
            inputs=[input_text],
            outputs=[output_label],
        )

    demo.launch()

if __name__ == "__main__":
    main()
