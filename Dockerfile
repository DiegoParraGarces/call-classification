# Imagen base con Python
FROM python:3.9-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo principal y otros necesarios al contenedor
COPY src/main.py ./main.py
COPY requirements.txt ./requirements.txt

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gradio

# Exponer el puerto utilizado por Gradio (7860 por defecto)
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Comando para ejecutar la aplicación
CMD ["python", "main.py"]

