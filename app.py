from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import os
import sqlite3
import secrets
import smtplib
from email.mime.text import MIMEText
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn.functional as F
from werkzeug.security import generate_password_hash, check_password_hash

from dotenv import load_dotenv


# === CONFIG ===
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY') # Change this in production!
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATABASE = 'database/users.db'

EMAIL_SENDER = os.getenv('EMAIL_SENDER')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')  # App password if Gmail 2FA

# ML Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_PATH = "ORAL_CANCER_AMPIRE/dataset"
MODEL_PATH = "static/model_file/oral_cancer_classifier.pth"
CLASS_NAMES = ['Abnormal', 'Normal', 'Null']

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# === DB Setup ===
def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_verified INTEGER DEFAULT 0,
            otp TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# === Helper Functions ===
def send_otp_email(receiver_email, otp):
    message = MIMEText(f"Your OTP code is: {otp}")
    message['Subject'] = 'Your OTP for Oral Cancer Detection Portal'
    message['From'] = EMAIL_SENDER
    message['To'] = receiver_email

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(message)

# === Routes ===

@app.route('/')
def home():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        otp = secrets.token_hex(3)  # 6-digit OTP

        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        try:
            c.execute('INSERT INTO users (email, password, otp) VALUES (?, ?, ?)', (email, password, otp))
            conn.commit()
            send_otp_email(email, otp)
            session['email_temp'] = email
            flash('OTP sent to your email!', 'info')
            return redirect(url_for('verify_otp'))
        except sqlite3.IntegrityError:
            flash('Email already exists.', 'danger')
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/verify', methods=['GET', 'POST'])
def verify_otp():
    if request.method == 'POST':
        user_otp = request.form['otp']
        email = session.get('email_temp')

        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute('SELECT otp FROM users WHERE email=?', (email,))
        real_otp = c.fetchone()
        if real_otp and real_otp[0] == user_otp:
            c.execute('UPDATE users SET is_verified=1, otp=NULL WHERE email=?', (email,))
            conn.commit()
            flash('Email verified! You can now log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Invalid OTP!', 'danger')
        conn.close()
    return render_template('verify_otp.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute('SELECT id, password, is_verified FROM users WHERE email=?', (email,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[1], password):
            if user[2] == 1:
                session['user_id'] = user[0]
                return redirect(url_for('home'))
            else:
                flash('Please verify your email first.', 'warning')
        else:
            flash('Incorrect credentials.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

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

if __name__ == '__main__':
    app.run(debug=True)
