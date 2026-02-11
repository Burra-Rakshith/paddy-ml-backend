from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'paddy_disease_model.h5')
CLASSES_PATH = os.path.join(BASE_DIR, 'model', 'classes.txt')
CONFIDENCE_THRESHOLD = 0.55

# Load model and classes
model = None
classes = []

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
else:
    print(f"Warning: Model not found at {MODEL_PATH}")

if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    print(f"Classes loaded: {classes}")
else:
    # Default classes if classes.txt is missing
    classes = [
        "Bacterial Leaf Blight",
        "Brown Spot",
        "Healthy Rice Leaf",
        "Leaf Blast",
        "Leaf scald",
        "Sheath Blight"
    ]

# Prediction Tips Metadata
DISEASE_INFO = {
    "Bacterial Leaf Blight": {
        "severity": "High",
        "tips": {
            "en": ["Use resistant varieties", "Avoid excess nitrogen", "Ensure good drainage"],
            "te": ["నిరోధక విత్తనాలు ఉపయోగించండి", "అధిక నత్రజని వాడకండి", "నీటి పారుదల సౌకర్యం మెరుగుపరచండి"]
        }
    },
    "Brown Spot": {
        "severity": "Medium",
        "tips": {
            "en": ["Use balanced fertilizers", "Treat seeds before sowing", "Maintain proper irrigation"],
            "te": ["సమతుల్య ఎరువులు వాడండి", "విత్తన శుద్ధి చేయండి", "సరైన నీటి పారుదల నిర్వహించండి"]
        }
    },
    "Healthy Rice Leaf": {
        "severity": "None",
        "tips": {
            "en": ["Continue regular monitoring", "Follow recommended farming practices"],
            "te": ["క్రమం తప్పకుండా పర్యవేక్షించండి", "సిఫార్సు చేసిన సాగు పద్ధతులను పాటించండి"]
        }
    },
    "Leaf Blast": {
        "severity": "High",
        "tips": {
            "en": ["Avoid excessive nitrogen", "Use fungicides if necessary", "Keep fields clean"],
            "te": ["అధిక నత్రజని వాడకండి", "అవసరమైతే శిలీంద్రనాశకాలు వాడండి", "పొలాలను శుభ్రంగా ఉంచండి"]
        }
    },
    "Leaf scald": {
        "severity": "Medium",
        "tips": {
            "en": ["Remove infected crop residues", "Use certified seeds", "Avoid overhead irrigation"],
            "te": ["వ్యాధి సోకిన పంట వ్యర్థాలను తొలగించండి", "ధృవీకరించబడిన విత్తనాలను వాడండి", "పైన నుండి నీరు చల్లడం నివారించండి"]
        }
    },
    "Sheath Blight": {
        "severity": "High",
        "tips": {
            "en": ["Maintain wider spacing between plants", "Control weeds", "Apply recommended fungicides"],
            "te": ["మొక్కల మధ్య తగిన దూరం పాటించండి", "కలుపు నియంత్రించండి", "సిఫార్సు చేసిన శిలీంద్రనాశకాలు వాడండి"]
        }
    },
    "Not a Paddy Leaf": {
        "severity": "N/A",
        "tips": {
            "en": ["The uploaded image does not appear to be a paddy leaf. Please upload a clear image of a paddy leaf."],
            "te": ["అప్‌లోడ్ చేసిన చిత్రం వరి ఆకులా లేదు. దయచేసి స్పష్టమైన వరి ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి."]
        }
    }
}

def is_paddy_leaf(img):
    """
    Simple heuristic to check if image has paddy leaf characteristics (Green/Yellow dominant).
    """
    try:
        # Convert to HSV
        img_hsv = img.convert('HSV')
        img_np = np.array(img_hsv)
        
        # Extract channels
        H = img_np[:,:,0]
        S = img_np[:,:,1]
        V = img_np[:,:,2]
        
        # Define Green/Yellow range (approximate for PIL HSV 0-255)
        # Green/Yellow hue is roughly 20 to 130
        # Saturation and Value should be sufficient to avoid black/white/gray
        leaf_mask = (H > 20) & (H < 130) & (S > 25) & (V > 25)
        
        leaf_pixels = np.sum(leaf_mask)
        total_pixels = img_np.shape[0] * img_np.shape[1]
        
        ratio = leaf_pixels / total_pixels
        print(f"Leaf pixel ratio: {ratio:.4f}")
        
        # Threshold: at least 10% of the image should be "leaf-colored"
        return ratio > 0.10
    except Exception as e:
        print(f"Error in heuristic check: {e}")
        return True # Fallback to model if check fails

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded on server."}
    
    # Read and preprocess image
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Check if it looks like a paddy leaf (Color Heuristic)
    if not is_paddy_leaf(img):
        return {
            "disease": "Not a Paddy Leaf",
            "confidence": 0.0,
            "severity": "N/A",
            "tips": DISEASE_INFO["Not a Paddy Leaf"]["tips"]
        }

    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][class_idx])
    disease_name = classes[class_idx]
    
    if confidence < CONFIDENCE_THRESHOLD:
        disease_name = "Not a Paddy Leaf"
    
    # Get additional info
    info = DISEASE_INFO.get(disease_name, {
        "severity": "Unknown",
        "tips": {"en": ["Consult an agricultural expert."], "te": ["వ్యవసాయ నిపుణుడిని సంప్రదించండి."]}
    })
    
    return {
        "disease": disease_name,
        "confidence": confidence,
        "severity": info["severity"],
        "tips": info["tips"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
