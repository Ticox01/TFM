import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import models
from collections import Counter
from torchvision.transforms import v2

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

import matplotlib.pyplot as plt

import numpy as np

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    @property
    def classes(self):
        return self.data.classes
    @property
    def get_labels_count(self):
        _, labels = zip(*self.data.imgs)
        return Counter(labels)

# Transformaciones de entrenamiento
train_transform = v2.Compose([
    v2.ToImage(),  # Convierte PIL → Tensor
    v2.RandomAffine(
        degrees=20,                     # Rotación ±20°
        translate=(0.2, 0.2),           # 20% de traslación horizontal y vertical
        scale=(0.9, 1.1)                # Zoom 0.9x–1.1x
    ),
    v2.RandomHorizontalFlip(p=0.5),     # Volteo horizontal 50%
    v2.ToDtype(torch.float32, scale=True),  # Convierte a float y divide entre 255
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

tensor_transform = v2.Compose([
    v2.ToImage(),  # Convierte PIL → Tensor
    v2.ToDtype(torch.float32,scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = ImageDataset("Datasets/FDAC_DATASET/train/", transform=train_transform)

train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)

valid_dataset = ImageDataset("Datasets/FDAC_DATASET/valid/", transform=tensor_transform)

valid_dataloader = DataLoader(valid_dataset, batch_size=10, shuffle=False)

test_dataset = ImageDataset("Datasets/FDAC_DATASET/test/", transform=tensor_transform)

test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

modelo= models.vgg19(weights="IMAGENET1K_V1")
modelo.classifier[6] = nn.Linear(4096, 2)

modelo = modelo.to("xpu")

for param in modelo.features.parameters():
    param.requires_grad = False

for param in modelo.features[24:].parameters():  # últimas capas convolucionales
    param.requires_grad = True

# TRAINING LOOP
# Loss function
criterion= nn.CrossEntropyLoss()
criterion = criterion.to("xpu")
optimizer= torch.optim.Adam(modelo.parameters(), lr=1e-4) #Adam suele ser el mejor para  la mayoría de los casos. 
EPOCHS= 100
setps_per_epoch = 150
patience = 5
patience_count = 0
best_val_loss = None

train_losses, val_losses= [], []

for epoch in range(EPOCHS):
    modelo.train(True) #Esto existe porque toma los métodos de la clase padre con super. Ponemos el modelo a entrenarse (iniciamos ese entrenamiento)
    running_loss= 0.0 # Track the loss as we train. 

    data_iter = iter(train_dataloader)
    for step in range(setps_per_epoch):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            # si se acabó el dataset, reiniciamos el iterador
            data_iter = iter(train_dataloader)
            images, labels = next(data_iter)
        images = images.to("xpu")
        labels = labels.to("xpu")
        optimizer.zero_grad()
        outputs = modelo(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    train_loss= running_loss / len(train_dataloader.dataset)
    train_losses.append(train_loss)

    #VALIDATION

    modelo.eval()
    running_loss= 0.0 #Miramos la loss que va teniendo la validación igual que el train. 
    with torch.no_grad():
        for data, labels in valid_dataloader:
            #Aquí no se optimiza ni se hace step puesto que es validación con lo que el modelo tiene ahora y ya entrenado, sin tocar nada. 
            data = data.to("xpu")
            labels = labels.to("xpu")
            outputs= modelo(data)
            loss= criterion(outputs, labels)
            running_loss += loss.item() * data.size(0)

    val_loss= running_loss / len(valid_dataloader.dataset)
    val_losses.append(val_loss)
    if best_val_loss is None:
        best_val_loss = val_loss
        patience_count = 0
    elif val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(modelo.state_dict(), 'best-model-parameters.pt')
        patience_count = 0
    else:
        patience_count += 1

    if patience_count > patience:
        print(f"Early Stoping in Epoch {epoch}")
        break

    print(f"Epoch {epoch+1}/{EPOCHS} - "
          f"Train loss: {train_loss:.4f}, "
          f"Val loss: {val_loss:.4f}")

# --------------------
# Test
# --------------------
# Asegúrate de que el modelo esté en modo evaluación
modelo.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_dataloader:
        
        batch_x = batch_x.to("xpu")
        batch_y = batch_y.to("xpu")
        outputs = modelo(batch_x)    
        # Predicción: índice de la clase con mayor score
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.append(preds.cpu())
        all_labels.append(batch_y.cpu())

# Concatenar todos los batches
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)


# ESTO ES PARA LA ACCURACY y f1 (te interesa que esta ultima sea buena)

acc= accuracy_score(all_labels, all_preds)
f1= f1_score(all_labels, all_preds, average= "macro")

print("f1:", f1, "\n", "Acc:", acc)


# ESTO ES PARA LA CM

cm = confusion_matrix(all_labels, all_preds)

# Visualización
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= ["autistic","non_autistic"])
disp.plot(cmap=plt.cm.Blues)
plt.show()