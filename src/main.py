import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class HuggingFaceModel:
    def __init__(self, model_name="jaortizb/modelo18"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probabilities.tolist()

# Función para la interfaz
def predict_with_model(input_text):
    model = HuggingFaceModel()
    predictions = model.predict(input_text)
    return predictions

# Definir interfaz en Gradio
def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Modelo Hugging Face en Gradio")
        input_text = gr.Textbox(label="Ingrese texto para predecir")
        output_label = gr.Label(label="Predicción")
        predict_button = gr.Button("Predecir")
        
        predict_button.click(
            fn=predict_with_model,
            inputs=[input_text],
            outputs=[output_label],
        )

    demo.launch()

if __name__ == "__main__":
    main()

