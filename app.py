import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from urllib.parse import quote as url_quote
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import base64
from pyngrok import ngrok
from io import BytesIO
import random

app = Flask(__name__)

# Coloca tu token de ngrok aquí
ngrok.set_auth_token("2no9UNajvQBnJaufy62CQNBJFzA_6cUymoP8rHMB1RDxG7fhq")

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
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Could not load image")

        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect facial landmarks
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            raise Exception("No face detected in the image")

        # Select 15 main keypoints
        key_points = [33, 133, 362, 263, 1, 61, 291, 199,
                     94, 0, 24, 130, 359, 288, 378]

        height, width = gray_image.shape

        # Function to add keypoints to image
        def add_keypoints_to_image(image, results, angle=0):
            plt.clf()
            fig = plt.figure(figsize=(8, 8))
            plt.imshow(image, cmap='gray')

            # Plot facial landmarks
            for point_idx in key_points:
                landmark = results.multi_face_landmarks[0].landmark[point_idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                # Rotate keypoints if necessary
                if angle != 0:
                    center = (width // 2, height // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated_point = np.dot(rotation_matrix[:, :2], np.array([x, y])) + rotation_matrix[:, 2]
                    x, y = rotated_point.astype(int)

                plt.plot(x, y, 'rx')

            # Save plot to memory
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            return base64.b64encode(buf.getvalue()).decode('utf-8')

        # Original image with keypoints
        original_image_base64 = add_keypoints_to_image(gray_image, results)

        # Flip image horizontally and add keypoints
        flipped_image = cv2.flip(gray_image, 1)
        flipped_image_base64 = add_keypoints_to_image(flipped_image, results)

        # Brighten image and add keypoints
        bright_image = np.clip(random.uniform(1.5, 2) * gray_image, 0, 255)
        bright_image_base64 = add_keypoints_to_image(bright_image.astype(np.uint8), results)

        # Rotate image by 180 degrees and add keypoints (adjust keypoints)
        rotated_image = cv2.rotate(gray_image, cv2.ROTATE_180)
        rotated_image_base64 = add_keypoints_to_image(rotated_image, results, angle=180)

        return {
            'original_image': original_image_base64,
            'flipped_image': flipped_image_base64,
            'bright_image': bright_image_base64,
            'rotated_image': rotated_image_base64
        }

    except Exception as e:
        print(f"Error in analyze_face: {str(e)}")
        raise

@app.route('/')
def home():
    images = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            images.append(filename)
    return render_template('index.html', images=images)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'File type not allowed'}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

        result_images = analyze_face(filepath)

        return jsonify(result_images)

    except Exception as e:
        print(f"Error in /analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Ejecuta la aplicación Flask
if __name__ == '__main__':
    public_url = ngrok.connect(5001)
    print(f" * ngrok URL: {public_url}")
    app.run(port=5001)
