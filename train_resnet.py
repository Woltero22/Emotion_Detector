import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
from IPython.display import FileLink


seed = torch.manual_seed(42)
train_dir = '/kaggle/input/raf-db-dataset/DATASET/train'
test_dir = '/kaggle/input/raf-db-dataset/DATASET/test'
num_epochs = 30


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=4),
}

class_names = image_datasets['train'].classes
print(f"Klasy emocji: {class_names}")

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    corrects = torch.sum(preds == labels.data)
    return corrects.item() / len(labels)


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corr = 0.0
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_corr += calculate_accuracy(outputs, labels)
        
    epoch_loss = running_loss / len(dataloaders['train'])
    epoch_acc = running_corr / len(dataloaders['train'])
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, Accuracy: {epoch_acc}")

    model.eval()
    val_loss = 0.0
    val_corr = 0.0
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_corr += calculate_accuracy(outputs, labels)
    val_loss /= len(dataloaders['test'])
    val_acc = val_corr / len(dataloaders['test'])

    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

print("Training complete.")

def get_model_parameters(model):
    return model.state_dict()

def get_confusion_matrix(model, dataloader):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return confusion_matrix(all_labels, all_preds)

def save_model(model, path):
    torch.save(model.state_dict(), path)

model_parameters = get_model_parameters(model)
conf_matrix = get_confusion_matrix(model, dataloaders['test'])
save_model(model, 'trained_model.pth')
download_link = FileLink('trained_model.pth')

print("Download link:", download_link)
print("Confusion Matrix:")
print(conf_matrix)