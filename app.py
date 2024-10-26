from flask import Flask, request, render_template
import cv2
import numpy as np
import dlib
import os
import base64
import requests

app = Flask(__name__)

def download_model():
    model_path = 'shape_predictor_68_face_landmarks.dat'
    if not os.path.exists(model_path):  # Verifica si el archivo ya existe
        print("Descargando el modelo...")
        url = "https://drive.google.com/file/d/1PM5qHgEOXRn4shStxYyvOvkuxEWUHH0s/view?usp=sharing"  # Reemplaza esto con tu enlace
        response = requests.get(url)
        with open(model_path, "wb") as file:
            file.write(response.content)
    else:
        print("Siiuuuuuu.")

# Llama a la función para descargar el modelo antes de usar el predictor
download_model()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

@app.route('/', methods=['GET', 'POST'])
def index():
    img_data = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No se encontró el archivo", 400

        file = request.files['image']

        if file.filename == '':
            return "No se seleccionó un archivo", 400

        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            puntos = [
                (landmarks.part(17).x, landmarks.part(17).y),
                (landmarks.part(21).x, landmarks.part(21).y),
                (landmarks.part(22).x, landmarks.part(22).y),
                (landmarks.part(26).x, landmarks.part(26).y),
                (landmarks.part(30).x, landmarks.part(30).y),
                (landmarks.part(36).x, landmarks.part(36).y),
                (landmarks.part(38).x, landmarks.part(38).y),
                (landmarks.part(39).x, landmarks.part(39).y),
                (landmarks.part(42).x, landmarks.part(42).y),
                (landmarks.part(43).x, landmarks.part(43).y),
                (landmarks.part(45).x, landmarks.part(45).y),
                (landmarks.part(51).x, landmarks.part(51).y),
                (landmarks.part(57).x, landmarks.part(57).y),
                (landmarks.part(60).x, landmarks.part(60).y),
                (landmarks.part(64).x, landmarks.part(64).y),
            ]

            for punto in puntos:
                size = 2  # Cambia este valor para aumentar el tamaño de las 'x'
                cv2.line(image, (punto[0] - size, punto[1] - size), (punto[0] + size, punto[1] + size), (0, 0, 255), 2)
                cv2.line(image, (punto[0] - size, punto[1] + size), (punto[0] + size, punto[1] - size), (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', image)
        img_str = base64.b64encode(buffer).decode('utf-8')

        return render_template('index.html', img_data=img_str)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
