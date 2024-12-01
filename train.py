import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_datasets_and_loaders
from model import EfficientNet
from tqdm import tqdm  # Import de tqdm pour la barre de progression

def train_one_epoch(model, train_loader, criterion, optimizer, device):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Boucle avec tqdm pour afficher une barre de progression
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        # Convertir les labels en indices pour CrossEntropyLoss
        labels = torch.argmax(labels, dim=1)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    accuracy = 100.0 * correct / total
    return epoch_loss, accuracy

def validate_one_epoch(model, val_loader, criterion, device):
    """
    Valide le modèle pendant une époque avec affichage de la progression.
    
    Args:
        model (nn.Module): Le modèle à valider.
        val_loader (DataLoader): DataLoader pour la validation.
        criterion (nn.Module): Fonction de perte.
        device (torch.device): CPU ou GPU.

    Returns:
        float: La perte moyenne.
        float: L'exactitude sur l'ensemble de validation.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        # Boucle avec tqdm pour afficher une barre de progression
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            # Convertir les labels en indices pour CrossEntropyLoss
            labels = torch.argmax(labels, dim=1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = 100.0 * correct / total
    return epoch_loss, accuracy

def main():
    # Configuration
    data_dir = "./processed_dataset"
    num_classes = 7
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 20
    model_save_path = "./efficientnet_model2.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger les datasets et DataLoaders
    train_loader, val_loader = get_datasets_and_loaders(data_dir, batch_size)

    # Charger le modèle
    model = EfficientNet(num_classes=num_classes, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2)
    model = model.to(device)

    # Définir la fonction de perte et l'optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Entraînement et validation
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Phase d'entraînement
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Phase de validation
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Sauvegarde du modèle après chaque époque
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
