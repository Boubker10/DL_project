import os
import subprocess

# Configuration
class Config:
    TENSORFLOW_MODEL_DIR = "./efficientnet_model_tf"  # Répertoire contenant le modèle TensorFlow SavedModel
    TENSORFLOWJS_MODEL_DIR = "./efficientnet_model_tfjs"  # Répertoire pour sauvegarder le modèle TensorFlow.js

# Vérifier si le répertoire TensorFlow.js existe
def check_and_create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Répertoire créé : {output_dir}")

# Convertir TensorFlow SavedModel en TensorFlow.js
def convert_tf_to_tfjs(tf_model_dir, tfjs_model_dir):
    try:
        print("Conversion du modèle TensorFlow en TensorFlow.js...")
        command = [
            "tensorflowjs_converter",
            "--input_format", "tf_saved_model",  # Format d'entrée
            "--output_format", "tfjs_graph_model",  # Format de sortie
            tf_model_dir,  # Chemin du modèle TensorFlow SavedModel
            tfjs_model_dir  # Chemin pour le modèle TensorFlow.js
        ]
        subprocess.run(command, check=True)
        print(f"Modèle TensorFlow.js sauvegardé dans : {tfjs_model_dir}")
    except Exception as e:
        print(f"Erreur lors de la conversion : {e}")

# Script principal
if __name__ == "__main__":
    # Vérification et création des répertoires nécessaires
    check_and_create_output_dir(Config.TENSORFLOWJS_MODEL_DIR)

    # Convertir le modèle
    convert_tf_to_tfjs(Config.TENSORFLOW_MODEL_DIR, Config.TENSORFLOWJS_MODEL_DIR)
