import os
from PIL import Image

def convert_images_to_rgb(input_dir, output_dir):
    """
    Convertit toutes les images dans le répertoire d'entrée en format RGB et les sauvegarde dans un répertoire de sortie.

    Args:
        input_dir (str): Chemin vers le dossier contenant les images d'entrée.
        output_dir (str): Chemin vers le dossier pour sauvegarder les images converties.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, relative_path)

            # Créer les dossiers de sortie si nécessaire
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            try:
                # Charger l'image
                with Image.open(input_path) as img:
                    # Convertir en RGB si nécessaire
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Sauvegarder l'image convertie
                    img.save(output_path)

                print(f"Converted: {input_path} -> {output_path}")

            except Exception as e:
                print(f"Erreur lors de la conversion de {input_path}: {e}")

if __name__ == "__main__":
    input_directory = "./dataset"  # Dossier contenant les images d'origine (train et validation)
    output_directory = "./processed_dataset"  # Dossier pour sauvegarder les images converties

    convert_images_to_rgb(input_directory, output_directory)
