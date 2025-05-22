from flask import Flask, render_template, request, redirect, jsonify
import os
import random
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn.functional as F
from pycocotools.coco import COCO

# === CONFIG ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_PATH = "/Users/hardikm-visiobyte/Desktop/Oral_Cancer_Ampire/dataset"
MODEL_PATH = "oral_cancer_classifier.pth"
CLASS_NAMES = ['Abnormal', 'Normal', 'Null']

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Load Model ===
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# === Flask App ===
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === Home page ===
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            label, confidence = predict_image(filepath)
            return render_template('result.html', label=label, confidence=confidence, image_path=filepath)
    
    return render_template('index.html')



# === New Predict Route for API ===
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')

    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    predicted_label = CLASS_NAMES[predicted.item()]
    confidence_score = round(confidence.item() * 100, 2)

    return jsonify({'label': predicted_label, 'confidence': confidence_score})


# === Predict Random Test Image ===
@app.route('/random')
def random_predict():
    test_dir = os.path.join(BASE_PATH, "Test")
    ann_path = os.path.join(test_dir, "_annotations.coco.json")
    coco = COCO(ann_path)
    img_ids = coco.getImgIds()
    img_id = random.choice(img_ids)
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(test_dir, img_info['file_name'])

    label, confidence = predict_image(img_path)
    return render_template('result.html', label=label, confidence=confidence, image_path=img_path)

# === Prediction Function ===
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    predicted_label = CLASS_NAMES[predicted.item()]
    confidence_score = confidence.item() * 100

    return predicted_label, round(confidence_score, 2)

# === Run the app ===
if __name__ == '__main__':
    app.run(debug=True)
