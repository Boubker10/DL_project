<<<<<<< HEAD
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from model import EfficientNet
from dataset import VeggieTorchDataset  


class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "./efficientnet_model2.pth"
    TEST_DATA_DIR = "./processed_dataset/test"  
    IMAGE_SIZE = 128
    BATCH_SIZE = 32
    CLASSES = ['carrot', 'eggplant', 'peas', 'potato', 'sweetcorn', 'tomato', 'turnip']


def load_model(model_path, num_classes, device):
    model = EfficientNet(num_classes=num_classes, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2)
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device).eval()
    return model

# Préparer les données de test
def get_test_loader(test_data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = VeggieTorchDataset(root_dir=test_data_dir, aug=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return test_loader

def test_model_with_confusion_matrix(model, test_loader, device):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = torch.argmax(labels, dim=1)  
            

            outputs = model(images)
            _, predicted = outputs.max(1)

 
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(Config.CLASSES))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Config.CLASSES)


    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    print("Chargement du modèle...")
    model = load_model(Config.MODEL_PATH, len(Config.CLASSES), Config.DEVICE)

    print("Préparation des données de test...")
    test_loader = get_test_loader(Config.TEST_DATA_DIR, Config.BATCH_SIZE)

    print("Test du modèle...")
    test_model_with_confusion_matrix(model, test_loader, Config.DEVICE)
=======
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from model import EfficientNet
from dataset import VeggieTorchDataset  


class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "./efficientnet_model2.pth"
    TEST_DATA_DIR = "./processed_dataset/test"  
    IMAGE_SIZE = 128
    BATCH_SIZE = 32
    CLASSES = ['carrot', 'eggplant', 'peas', 'potato', 'sweetcorn', 'tomato', 'turnip']


def load_model(model_path, num_classes, device):
    model = EfficientNet(num_classes=num_classes, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2)
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device).eval()
    return model

# Préparer les données de test
def get_test_loader(test_data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = VeggieTorchDataset(root_dir=test_data_dir, aug=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return test_loader

def test_model_with_confusion_matrix(model, test_loader, device):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = torch.argmax(labels, dim=1)  
            

            outputs = model(images)
            _, predicted = outputs.max(1)

 
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(Config.CLASSES))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Config.CLASSES)


    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    print("Chargement du modèle...")
    model = load_model(Config.MODEL_PATH, len(Config.CLASSES), Config.DEVICE)

    print("Préparation des données de test...")
    test_loader = get_test_loader(Config.TEST_DATA_DIR, Config.BATCH_SIZE)

    print("Test du modèle...")
    test_model_with_confusion_matrix(model, test_loader, Config.DEVICE)
>>>>>>> 0cb6f866422f743838d1f6eb3c3d883273399d1c
