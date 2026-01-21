from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import mysql.connector
import numpy as np
import tensorflow as tf
import base64
import io
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash
import os
from dotenv import load_dotenv
load_dotenv()
password = os.getenv('password')
app = Flask(__name__)
app.secret_key = 'your_secret_key'
model = tf.keras.models.load_model("emnist_balanced_model.h5")
label_map = {}
with open('emnist-balanced-mapping.txt') as f:
    for line in f:
        key, val = line.strip().split()
        label_map[int(key)] = chr(int(val))
        
db = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password=password,  
    database="smartdigitapp"
)
cursor = db.cursor()
@app.route('/')
def home():
    if 'user_id' in session:
        return render_template('index.html')
    return redirect('/login')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        cursor.execute("SELECT id, password_hash FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()
        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            return redirect('/')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        cursor.execute("INSERT INTO users (email, password_hash) VALUES (%s, %s)", (email, password))
        db.commit()
        return redirect('/login')
    return render_template('register.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect('/login')


def center_image(img_array):
    img_array = (img_array > 50) * 255
    coords = np.argwhere(img_array)
    if coords.any():
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        cropped = img_array[y0:y1, x0:x1]
        cropped_img = Image.fromarray(cropped.astype(np.uint8)).resize((20, 20), Image.LANCZOS)
        new_img = Image.new('L', (28, 28), color=0)
        new_img.paste(cropped_img, ((28 - 20) // 2, (28 - 20) // 2))
        return np.array(new_img)
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image']

        mode = data.get('mode', 'auto') 

        if mode == 'digit':
            allowed_labels = list(range(10))
        elif mode == 'letter':
            allowed_labels = [k for k, v in label_map.items() if v.isalpha()]
        else:
            allowed_labels = list(label_map.keys())

        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]
        image_data += '=' * (-len(image_data) % 4)

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = Image.eval(image, lambda x: 255 - x)
        image = image.resize((28, 28))
        img_array = np.array(image)
        centered_img = center_image(img_array)
        centered_img = centered_img / 255.0
        input_data = centered_img.reshape(1, 28, 28, 1)
        Image.fromarray((centered_img.reshape(28, 28) * 255).astype(np.uint8)).save("input_check.png")


        prediction = model.predict(input_data)[0] 

        filtered_probs = {k: prediction[k] for k in allowed_labels}
        predicted_label = max(filtered_probs, key=filtered_probs.get)
        predicted_char = label_map[predicted_label]
        confidence = float(filtered_probs[predicted_label]) * 100

        if 'user_id' in session:
            cursor.execute(
                "INSERT INTO predictions (user_id, predicted_char, confidence) VALUES (%s, %s, %s)",
                (session['user_id'], predicted_char, round(confidence, 2))
            )
            db.commit()

        return {'character': predicted_char, 'confidence': round(confidence, 2)}

    except Exception as e:
        print("Prediction error:", e)
        return {'error': 'Prediction failed'}, 500


@app.route('/history')
def history():
    cursor.execute(
        "SELECT predicted_char, confidence, predicted_at FROM predictions WHERE user_id=%s ORDER BY predicted_at DESC",
        (session['user_id'],)
    )
    records = cursor.fetchall()
    return render_template('history.html', records=records)


@app.route('/clear_history', methods=['POST'])
def clear_history():
    if 'user_id' in session:
        cursor.execute(
            "DELETE FROM predictions WHERE user_id=%s",
            (session['user_id'],)
        )
        db.commit()
    return redirect('/history')


if __name__ == '__main__':
    app.run(debug=True)
