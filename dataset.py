import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import List  # Importer List pour les annotations de type

class VeggieTorchDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, aug=None) -> None:
        """
        Dataset personnalisé pour les classes de légumes.
        
        Args:
            root_dir (str): Chemin racine contenant les sous-dossiers pour les classes.
            aug (torchvision.transforms.Compose, optional): Transformations d'augmentation et de prétraitement.
        """
        self.dataset = []
        self.root_dir = root_dir
        
        # Ajouter les classes de légumes et leurs vecteurs one-hot
        self.__add_dataset__("carrot", [1, 0, 0, 0, 0, 0, 0])
        self.__add_dataset__("eggplant", [0, 1, 0, 0, 0, 0, 0])
        self.__add_dataset__("peas", [0, 0, 1, 0, 0, 0, 0])
        self.__add_dataset__("potato", [0, 0, 0, 1, 0, 0, 0])
        self.__add_dataset__("sweetcorn", [0, 0, 0, 0, 1, 0, 0])
        self.__add_dataset__("tomato", [0, 0, 0, 0, 0, 1, 0])
        self.__add_dataset__("turnip", [0, 0, 0, 0, 0, 0, 1])

        # Transformation par défaut si aucune n'est fournie
        if aug is None:
            self.augmentation = transforms.Compose([
                transforms.Resize((150, 150)),
                transforms.CenterCrop((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.augmentation = aug
    
    def __add_dataset__(self, dir_name: str, class_label: List[int]) -> None:
        """
        Ajoute les images d'une classe au dataset.
        
        Args:
            dir_name (str): Nom du dossier de classe.
            class_label (List[int]): One-hot vector représentant la classe.
        """
        full_path = os.path.join(self.root_dir, dir_name)
        label = np.array(class_label)
        for fname in os.listdir(full_path):
            fpath = os.path.join(full_path, fname)
            fpath = os.path.abspath(fpath)
            self.dataset.append((fpath, label))

    def __len__(self) -> int:
        """
        Retourne la taille du dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index: int):
        """
        Retourne un élément du dataset (image, label).
        
        Args:
            index (int): Index de l'élément.
        
        Returns:
            torch.Tensor: Image transformée.
            torch.Tensor: Label en one-hot.
        """
        fpath, label = self.dataset[index]

        # Charger l'image sous forme de PIL.Image
        image = Image.open(fpath)
        
        # Convertir toutes les images en RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Appliquer les transformations
        image = self.augmentation(image)

        # Convertir le label en tensor
        label = torch.tensor(label, dtype=torch.float32)

        return image, label


def get_datasets_and_loaders(data_dir, batch_size=32):
        """
        Charge les datasets et retourne les DataLoaders pour entraînement et validation.
        
        Args:
            data_dir (str): Chemin contenant les sous-dossiers 'train' et 'validation'.
            batch_size (int): Taille des batchs.
        
        Returns:
            tuple: (train_loader, val_loader)
        """
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'validation')

        # Création des datasets
        train_dataset = VeggieTorchDataset(train_dir)
        val_dataset = VeggieTorchDataset(val_dir)

        # Création des DataLoaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        return train_loader, val_loader


if __name__ == "__main__":
    # Exemple d'utilisation
    data_dir = "./dataset"  # Dossier contenant 'train' et 'validation'
    batch_size = 32

    train_loader, val_loader = get_datasets_and_loaders(data_dir, batch_size)

    print(f"Nombre d'exemples dans l'entraînement : {len(train_loader.dataset)}")
    print(f"Nombre d'exemples dans la validation : {len(val_loader.dataset)}")

    # Affichage d'un batch d'entraînement
    for images, labels in train_loader:
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        print("Label (one-hot):", labels[0])
        break
