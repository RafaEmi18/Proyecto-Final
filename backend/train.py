import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt

# Asegura uso de CPU o GPU si tienes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformaciones para normalizar, redimensionar y aumentar datos
transform = transforms.Compose([
    transforms.Resize((64, 64)),   # Tamaño fijo
    transforms.Grayscale(num_output_channels=1),  # Escala de grises
    transforms.RandomRotation(10),                # Rotación aleatoria
    transforms.RandomHorizontalFlip(),            # Volteo horizontal aleatorio
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Variación de brillo/contraste
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Cargar dataset
dataset = ImageFolder(root='braille_dataset', transform=transform)

# Separar en entrenamiento y validación
val_ratio = 0.2
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

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
        self.fc2 = nn.Linear(64, len(class_names))

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

# Entrenamiento y validación
epochs = 15
train_losses, val_losses, train_accs, val_accs = [], [], [], []
for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validación
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Época {epoch+1}/{epochs} | Pérdida: {train_loss:.4f} | Precisión: {train_acc:.2f}% | Val_Pérdida: {val_loss:.4f} | Val_Precisión: {val_acc:.2f}%")

print('Entrenamiento terminado.')

# Guardar el modelo
torch.save(model.state_dict(), 'braille_model.pth')
print('Modelo guardado como braille_model.pth')

# Graficar pérdidas y precisión
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Entrenamiento')
plt.plot(val_losses, label='Validación')
plt.title('Pérdida')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.subplot(1,2,2)
plt.plot(train_accs, label='Entrenamiento')
plt.plot(val_accs, label='Validación')
plt.title('Precisión (%)')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.tight_layout()
plt.savefig('training_metrics.png')
print('Gráficas guardadas como training_metrics.png')


