import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import torch
import numpy as np
from PIL import Image
from model import EfficientNet  # Assurez-vous que le modèle EfficientNet est bien importé
from torchvision import transforms
import base64
from io import BytesIO

# Configuration de l'application
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Les classes utilisées pendant l'entraînement
    CLASSES = ['carrot', 'eggplant', 'peas', 'potato', 'sweetcorn', 'tomato', 'turnip']
    IMAGE_SIZE = 128
    MODEL_PATH = "efficientnet_model.pth"

# Charger le modèle entraîné
def load_model(model_path, num_classes, device):
    try:
        model = EfficientNet(num_classes=num_classes, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2)
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device).eval()
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return None

# Transformations pour les images
transform = transforms.Compose([
    transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Charger le modèle
model = load_model(Config.MODEL_PATH, len(Config.CLASSES), Config.DEVICE)

# Vérification si le modèle est chargé
if model is None:
    raise RuntimeError("Le modèle n'a pas pu être chargé. Vérifiez le chemin ou le fichier du modèle.")

# Application Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Vegetable Classifier", className="text-center text-primary mb-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.H5("Upload an image of a vegetable:", className="text-primary"),
            dcc.Upload(
                id='upload-image',
                children=html.Div(['📤 Drag and Drop or ', html.A('Select Files')]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '10px',
                    'textAlign': 'center',
                    'margin': '10px',
                    'backgroundColor': '#f8f9fa',
                    'color': '#6c757d'
                },
                accept='image/*'
            ),
            html.Div(id='output-image-upload', className="mt-4"),
        ], width=6),
        dbc.Col([
            html.H5("Prediction and Probabilities:", className="text-primary"),
            html.Div(id='prediction-output', className="mt-4"),
            html.Div(id='probability-bars', className="mt-4"),
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col(html.Footer("© 2024 Vegetable Classifier", className="text-center text-muted mt-4"), width=12)
    ])
])

# Pipeline de prédiction
def predict(image):
    try:
        # Préparer l'image
        image = Image.open(image).convert('RGB')
        image = transform(image).unsqueeze(0).to(Config.DEVICE)

        # Prédictions
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy().flatten()
            predicted_class = Config.CLASSES[np.argmax(probabilities)]

        return predicted_class, probabilities
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        return None, None

# Créer des barres de probabilités
def create_probability_bars(probabilities):
    max_index = np.argmax(probabilities)
    bars = []
    for i, (cls, prob) in enumerate(zip(Config.CLASSES, probabilities)):
        is_max = (i == max_index)
        bars.append(
            html.Div(
                style={
                    'border': '2px solid #007bff' if is_max else '1px solid #007bff',
                    'borderRadius': '10px',
                    'padding': '10px',
                    'marginBottom': '15px',
                    'backgroundColor': '#f8f9fa',
                },
                children=[
                    html.Div(
                        style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'},
                        children=[
                            html.Div(
                                f"{cls.capitalize()}",
                                style={
                                    'width': '20%',
                                    'fontWeight': 'bold' if is_max else 'normal',
                                    'fontSize': '14px',
                                    'color': '#007bff' if is_max else 'black',
                                }
                            ),
                            html.Div(
                                style={
                                    'width': '60%',
                                    'height': '15px',
                                    'backgroundColor': '#e9ecef',
                                    'borderRadius': '7px',
                                    'position': 'relative',
                                    'overflow': 'hidden',
                                },
                                children=[
                                    html.Div(
                                        style={
                                            'width': f'{prob * 100:.2f}%',
                                            'height': '100%',
                                            'backgroundColor': '#007bff',
                                            'position': 'absolute',
                                            'borderRadius': '7px',
                                            'transition': 'width 0.5s ease-in-out',
                                        }
                                    )
                                ]
                            ),
                            html.Div(
                                f"{prob:.2%}",
                                style={
                                    'width': '20%',
                                    'textAlign': 'right',
                                    'fontSize': '14px',
                                    'fontWeight': 'bold' if is_max else 'normal',
                                    'color': '#007bff' if is_max else 'black',
                                }
                            )
                        ]
                    )
                ]
            )
        )
    return bars

# Callback pour mise à jour des prédictions
@app.callback(
    [Output('output-image-upload', 'children'),
     Output('prediction-output', 'children'),
     Output('probability-bars', 'children')],
    [Input('upload-image', 'contents')],
    [State('upload-image', 'filename')]
)
def update_output(content, filename):
    if content is not None:
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        image = BytesIO(decoded)

        # Prédictions
        predicted_class, probabilities = predict(image)
        if probabilities is None:
            return "Erreur dans la prédiction", None, None

        probability_bars = create_probability_bars(probabilities)

        return (
            html.Img(src=content, style={
                'maxWidth': '100%',
                'height': 'auto',
                'borderRadius': '10px',
                'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)'
            }),
            html.Div([
                html.H4("Predicted Class: ", className="text-secondary", style={'display': 'inline'}),
                html.H4(predicted_class.capitalize(), className="text-success", style={'display': 'inline', 'fontWeight': 'bold'})
            ]),
            probability_bars
        )
    else:
        return None, None, None

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
