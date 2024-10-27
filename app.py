import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import base64
import dlib
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Set backend before importing pyplot

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_face(image_path):
    try:
        # Initialize dlib detector and predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Could not load image")

        # Convert to grayscale for dlib
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray_image)

        if len(faces) == 0:
            raise Exception("No face detected in the image")

        # Select 15 main keypoints (same points as in your original dlib code)
        key_points_indices = [17, 21, 22, 26, 30, 36, 38, 39, 42, 43, 45, 51, 57, 60, 64]

        height, width = gray_image.shape

        # Create a new figure for each analysis
        plt.clf()
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(gray_image, cmap='gray')

        for face in faces:
            landmarks = predictor(gray_image, face)

            # Plot facial landmarks
            for idx in key_points_indices:
                x = int(landmarks.part(idx).x)
                y = int(landmarks.part(idx).y)
                plt.plot(x, y, 'rx')  # Draw red points

        # Save plot to memory
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        # Convert to base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return image_base64

    except Exception as e:
        print(f"Error in analyze_face: {str(e)}")
        raise
    finally:
        plt.close('all')

@app.route('/')
def home():
    # Get list of images in upload folder
    images = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            images.append(filename)
    return render_template('index.html', images=images)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Check if we're analyzing an existing file
        if 'existing_file' in request.form:
            filename = request.form['existing_file']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(filepath):
                return jsonify({'error': f'File not found: {filename}'}), 404
            
        # Check if we're uploading a new file
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'File type not allowed'}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        
        else:
            return jsonify({'error': 'No file provided'}), 400

        # Analyze the image
        result_image = analyze_face(filepath)
        
        return jsonify({
            'success': True,
            'image': result_image
        })

    except Exception as e:
        print(f"Error in /analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/delete', methods=['POST'])
def delete_image():
    try:
        filename = request.form['filename']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'success': True, 'message': 'Imagen eliminada con éxito.'})
        else:
            return jsonify({'success': False, 'message': 'Archivo no encontrado.'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# Ejecuta la aplicación Flask
if __name__ == '__main__':
    app.run(port=5000)
