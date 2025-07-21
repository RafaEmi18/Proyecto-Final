import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
import os
import time

CLASSES = ['A', 'B', 'C']

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

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

# Cargar el modelo
model = BrailleCNN()
model.load_state_dict(torch.load('braille_model.pth', map_location=torch.device('cpu')))
model.eval()

cap = cv2.VideoCapture(0)

# Asegurarse de que las carpetas para guardar imágenes existen
os.makedirs('braille_dataset/A', exist_ok=True)
os.makedirs('braille_dataset/B', exist_ok=True)
os.makedirs('braille_dataset/C', exist_ok=True)

print("Presiona 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Este es el cuadrito que aparece en la camara
    roi = frame[80:400, 160:480]  # Se puede ajustar
    image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    pil_image = transforms.ToPILImage()(image)
    input_tensor = transform(pil_image).unsqueeze(0) 

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    predicted_letter = CLASSES[int(pred.item())]
    confidence_value = confidence.item()

    text = f'{predicted_letter} ({confidence_value:.2f})'
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.rectangle(frame, (160, 80), (480, 400), (255, 0, 0), 2)
    cv2.imshow('Reconocimiento de Braille', frame)

    # Si la predicción tiene alta confianza, guarda la imagen en la carpeta correspondiente
    if confidence_value > 0.9:
        folder = f'braille_dataset/{predicted_letter}'
        filename = os.path.join(folder, f'{int(time.time())}.png')
        cv2.imwrite(filename, roi)  # Guardar la imagen completa, no solo el recorte
        print(f'Imagen guardada en: {filename}')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
