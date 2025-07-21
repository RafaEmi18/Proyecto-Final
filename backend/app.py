from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
import os

CLASSES = ['A', 'B', 'C']

# Modelo CNN de Braille
class BrailleCNN(nn.Module):
    def __init__(self):
        super(BrailleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, len(CLASSES))

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Crear la app Flask
app = Flask(__name__)
CORS(app, origins=['*'])  # Habilitar CORS para producción

# Cargar el modelo entrenado
model = BrailleCNN()
model.load_state_dict(torch.load('braille_model.pth', map_location=torch.device('cpu')))
model.eval()

# Transformaciones para la imagen
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route("/")
def home():
    return "<h1>¡Backend funcionando correctamente!</h1>"

# Ruta principal para recibir la imagen y hacer la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener la imagen en base64 desde el request
    data = request.get_json()
    img_data = data['image']
    img_data = base64.b64decode(img_data)
    
    # Convertir la imagen a formato PIL
    img = Image.open(io.BytesIO(img_data))

    # Preprocesar la imagen
    input_tensor = transform(img)
    if not torch.is_tensor(input_tensor):
        raise ValueError("La transformación no devolvió un tensor")
    input_tensor = input_tensor.unsqueeze(0)  # Agregar batch dimension (1, 1, 64, 64)

    # Realizar la predicción
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    predicted_letter = CLASSES[int(pred.item())]
    confidence_value = confidence.item()

    # Devolver la letra y la confianza
    return jsonify({
        'letter': predicted_letter,
        'confidence': confidence_value
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
