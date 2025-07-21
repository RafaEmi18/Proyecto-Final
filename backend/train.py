import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt

# Asegura uso de CPU o GPU si tienes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformaciones para normalizar y redimensionar
transform = transforms.Compose([
    transforms.Resize((64, 64)),   # Tamaño fijo
    transforms.Grayscale(num_output_channels=1),  # Convertir a escala de grises
    transforms.ToTensor(),        # A tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalización básica
])

# Cargar dataset
dataset = ImageFolder(root='braille_dataset', transform=transform)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Mapear clases
class_names = dataset.classes
print(f'Clases detectadas: {class_names}')

# Red neuronal CNN simple
class BrailleCNN(nn.Module):
    def __init__(self):
        super(BrailleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, len(class_names))  # salida depende del número de clases

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instanciar modelo
model = BrailleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Época {epoch+1}/{epochs}, Pérdida: {running_loss/len(train_loader):.4f}")

print('Entrenamiento terminado.')

# Guardar el modelo
torch.save(model.state_dict(), 'braille_model.pth')
print('Modelo guardado como braille_model.pth')


