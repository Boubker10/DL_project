import onnx
from onnx_tf.backend import prepare
import os
import tensorflow as tf

# Configuration
class Config:
    ONNX_MODEL_PATH = "./efficientnet_model.onnx"  
    TENSORFLOW_MODEL_DIR = "./efficientnet_model_tf" 


def check_onnx_model(onnx_model_path):
    try:
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        print("Le modèle ONNX est valide.")
        return onnx_model
    except Exception as e:
        print(f"Erreur lors de la vérification du modèle ONNX : {e}")
        return None


def convert_onnx_to_tf(onnx_model, tf_model_dir):
    try:
        print("Conversion du modèle ONNX en TensorFlow...")
        tf_rep = prepare(onnx_model)  
        tf_rep.export_graph(tf_model_dir)  
        print(f"Modèle TensorFlow sauvegardé dans : {tf_model_dir}")
    except Exception as e:
        print(f"Erreur lors de la conversion : {e}")


def test_tf_model(tf_model_dir, input_shape):
    try:
        print("Chargement et test du modèle TensorFlow...")
        model = tf.saved_model.load(tf_model_dir)
        infer = model.signatures["serving_default"]

        input_tensor = tf.random.normal(input_shape)
        output = infer(input=input_tensor)

        print(f"Sortie du modèle TensorFlow : {output}")
    except Exception as e:
        print(f"Erreur lors du test du modèle TensorFlow : {e}")

if __name__ == "__main__":
    onnx_model = check_onnx_model(Config.ONNX_MODEL_PATH)
    if onnx_model is None:
        exit("Modèle ONNX invalide, arrêt de la conversion.")
    convert_onnx_to_tf(onnx_model, Config.TENSORFLOW_MODEL_DIR)

    test_tf_model(Config.TENSORFLOW_MODEL_DIR, input_shape=(1, 3, 128, 128))  # Exemple : entrée RGB 128x128
