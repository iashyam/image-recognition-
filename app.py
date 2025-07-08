from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from main import recoganize_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Dummy label prediction function (replace with your model logic)
def predict_label(image_path):
    from PIL import Image
    image = Image.open(image_path)
    return recoganize_image(image)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    label = predict_label(filepath)
    return jsonify({'label': label})

if __name__ == '__main__':
    app.run(debug=True)
