import torch
from model import EfficientNet  

# Configuration
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "./efficientnet_model2.pth"  
    ONNX_PATH = "./efficientnet_model.onnx"  
    NUM_CLASSES = 7 
    IMAGE_SIZE = 128  

def load_model(model_path, num_classes, device):
    model = EfficientNet(num_classes=num_classes, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2)
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device).eval()
    return model


def export_to_onnx(model, onnx_path, image_size, device):
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)  
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,  
            opset_version=11,  
            do_constant_folding=True,  
            input_names=['input'], 
            output_names=['output'], 
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  
        )
        print(f"Modèle exporté avec succès en ONNX vers {onnx_path}")
    except Exception as e:
        print(f"Erreur lors de l'exportation en ONNX : {e}")

# Vérifier le modèle ONNX
def verify_onnx_model(onnx_path):
    import onnx
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("Le modèle ONNX est valide.")
    except Exception as e:
        print(f"Erreur lors de la vérification du modèle ONNX : {e}")

if __name__ == "__main__":
    print("Chargement du modèle PyTorch...")
    model = load_model(Config.MODEL_PATH, Config.NUM_CLASSES, Config.DEVICE)

    print("Exportation vers ONNX...")
    export_to_onnx(model, Config.ONNX_PATH, Config.IMAGE_SIZE, Config.DEVICE)

    print("Vérification du modèle ONNX...")
    verify_onnx_model(Config.ONNX_PATH)
