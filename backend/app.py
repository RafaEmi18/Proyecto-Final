from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
import os
import time
from statistical_estimation import BrailleStatisticalEstimator

CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

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
# Habilitar CORS: permitir dominio de producción y localhost durante desarrollo
allowed_origins = [
    'https://iimblsm-translator-frontend.onrender.com',
    'http://localhost:4321'
]
# Permitir añadir orígenes adicionales mediante la variable de entorno ALLOWED_ORIGINS
env_origins = os.environ.get('ALLOWED_ORIGINS')
if env_origins:
    allowed_origins.extend([o.strip() for o in env_origins.split(',') if o.strip()])

# Habilitar CORS en la aplicación
CORS(app, origins=allowed_origins, supports_credentials=True)

# Inicializar el estimador estadístico
estimator = BrailleStatisticalEstimator(confidence_level=0.95)

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
    start_time = time.time()
    
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
    response_time = time.time() - start_time
    
    # Obtener la letra real si está disponible (para evaluación)
    true_letter = data.get('true_letter', None)
    
    # Agregar la predicción al estimador estadístico
    if true_letter:
        estimator.add_prediction(predicted_letter, true_letter, confidence_value, response_time)

    # Devolver la letra y la confianza
    return jsonify({
        'letter': predicted_letter,
        'confidence': confidence_value,
        'response_time': response_time
    })

# Ruta para obtener estimaciones estadísticas
@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Obtener todas las estimaciones estadísticas"""
    estimates = estimator.get_comprehensive_estimates()
    return jsonify(estimates)

# Ruta para obtener resumen de estadísticas
@app.route('/statistics/summary', methods=['GET'])
def get_statistics_summary():
    """Obtener resumen de estadísticas en formato legible"""
    estimates = estimator.get_comprehensive_estimates()
    
    summary = {
        'overall_accuracy': {
            'point_estimate': f"{estimates['overall_accuracy']['point_estimate']:.3f} ({estimates['overall_accuracy']['point_estimate']*100:.1f}%)",
            'confidence_interval': f"[{estimates['overall_accuracy']['confidence_interval'][0]:.3f}, {estimates['overall_accuracy']['confidence_interval'][1]:.3f}]",
            'sample_size': estimates['overall_accuracy']['sample_size']
        },
        'overall_confidence': {
            'point_estimate': f"{estimates['overall_confidence']['point_estimate']:.3f} ({estimates['overall_confidence']['point_estimate']*100:.1f}%)",
            'confidence_interval': f"[{estimates['overall_confidence']['confidence_interval'][0]:.3f}, {estimates['overall_confidence']['confidence_interval'][1]:.3f}]",
            'sample_size': estimates['overall_confidence']['sample_size']
        },
        'response_time': {
            'point_estimate': f"{estimates['response_time']['point_estimate']:.3f} segundos",
            'confidence_interval': f"[{estimates['response_time']['confidence_interval'][0]:.3f}, {estimates['response_time']['confidence_interval'][1]:.3f}] segundos",
            'sample_size': estimates['response_time']['sample_size']
        },
        'total_predictions': estimates['metadata']['total_predictions']
    }
    
    return jsonify(summary)

# Ruta para guardar estimaciones
@app.route('/statistics/save', methods=['POST'])
def save_statistics():
    """Guardar las estimaciones actuales en un archivo"""
    try:
        filename = request.json.get('filename', None)
        estimator.save_estimates(filename)
        return jsonify({'message': 'Estimaciones guardadas exitosamente', 'filename': filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ruta para agregar predicción manual (para testing)
@app.route('/statistics/add-prediction', methods=['POST'])
def add_prediction():
    """Agregar una predicción manual para análisis estadístico"""
    try:
        data = request.json
        predicted_letter = data['predicted_letter']
        true_letter = data['true_letter']
        confidence = data['confidence']
        response_time = data.get('response_time', 0.0)
        
        estimator.add_prediction(predicted_letter, true_letter, confidence, response_time)
        
        return jsonify({'message': 'Predicción agregada exitosamente'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
