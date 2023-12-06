from flask import Flask, jsonify, request
import firebase_admin
from firebase_admin import credentials, storage
import cv2
import numpy as np
import dlib
import traceback
import os
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

cred = credentials.Certificate("C:/Users/Erick/Documents/FaceRecognition/uniface-9cd0b-firebase-adminsdk-n0htw-229f9c1c57.json")
firebase_admin.initialize_app(cred, {"storageBucket": "uniface-9cd0b.appspot.com"})
bucket = storage.bucket()

rostros_data = []
detector = dlib.get_frontal_face_detector()
shape_predictor_path = "C:/Users/Erick/Documents/FaceRecognition/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "C:/Users/Erick/Documents/FaceRecognition/dlib_face_recognition_resnet_model_v1.dat"

sp = dlib.shape_predictor(shape_predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)


def cargar_datos_de_rostros():
    try:
        blobs = bucket.list_blobs()
        for blob in blobs:
            image_file = blob.download_as_bytes()
            image_np = np.frombuffer(image_file, dtype=np.uint8)
            image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            rostros_data.append({"nombre": blob.name, "descriptor": obtener_descriptor_facial(image_cv)})
    except Exception as e:
        print('Error obteniendo datos de Firebase Storage:', str(e))


def obtener_descriptor_facial(imagen_cv):
    rects = detector(imagen_cv, 1)
    if len(rects) == 0:
        return None
    shape = sp(imagen_cv, rects[0])
    descriptor_facial = facerec.compute_face_descriptor(imagen_cv, shape)
    return descriptor_facial


def comparar_rostro(descriptor_input):
    for rostro in rostros_data:
        if rostro["descriptor"] is not None:
            distancia = np.linalg.norm(np.array(rostro["descriptor"]) - np.array(descriptor_input))
            if distancia < 0.6:
                return True, rostro["nombre"]
    return False, None


def es_formato_valido(filename):
    # Verificar si la extensión del archivo es válida
    allowed_extensions = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/', methods=['GET'])
def get_storage_data():
    try:
        blobs = bucket.list_blobs()
        storage_data = [{"name": blob.name,
                         "link": f"https://firebasestorage.googleapis.com/v0/b/uniface-9cd0b.appspot.com/o/{blob.name}?alt=media&token={blob.public_url.split('=')[-1]}"}
                        for blob in blobs if blob.name.lower().endswith(('.jpg', '.jpeg', '.png'))]
        return jsonify(storage_data)
    except Exception as e:
        print('Error obteniendo datos de Firebase Storage:', str(e))
        return jsonify({"error": "Internal Server Error"}), 500


@app.route('/api/compararRostro', methods=['POST'])
def comparar_rostro_route():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No se proporcionó ninguna imagen."}), 400
        image_file = request.files['image']
        image_np = np.frombuffer(image_file.read(), dtype=np.uint8)
        image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        descriptor_input = obtener_descriptor_facial(image_cv)
        if descriptor_input is not None:
            resultado_comparacion, nombre_similar = comparar_rostro(descriptor_input)
            if resultado_comparacion:
                return jsonify({"encontrado": True, "nombre": nombre_similar})
            else:
                return jsonify({"encontrado": False})
        else:
            return jsonify({"error": "No se detectaron rostros en la imagen de entrada."}), 400
    except Exception as e:
        print('Error en la comparación de rostros:', str(e))
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error"}), 500


@app.route('/api/insertarRostro', methods=['POST'])
def insertar_rostro_route():
    try:
        # Verificar si se proporcionaron imágenes
        if 'image' not in request.files:
            return jsonify({"error": "No se proporcionó ninguna imagen."}), 400

        # Obtener la lista de imágenes
        images = request.files.getlist('image')

        # Verificar si el formato de las imágenes es válido
        for image_file in images:
            if not es_formato_valido(image_file.filename):
                return jsonify({"error": f"Formato de imagen no válido. Solo se admiten archivos jpg, jpeg y png. ({image_file.filename})"}), 400

        # Procesar cada imagen
        for image_file in images:
            image_np = np.frombuffer(image_file.read(), dtype=np.uint8)
            image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            # Obtener el descriptor facial
            descriptor_input = obtener_descriptor_facial(image_cv)

            if descriptor_input is not None:
                # Cambiar 'nombre_de_la_imagen' por un nombre único o algún identificador deseado
                nombre_de_la_imagen = f"nombre_de_la_imagen_{os.path.splitext(image_file.filename)[0]}_{str(time.time())}.jpeg"

                # Subir la imagen a Firebase Storage
                blob = bucket.blob(nombre_de_la_imagen)
                blob.upload_from_string(cv2.imencode('.jpeg', image_cv)[1].tobytes(), content_type='image/jpeg')

                # Agregar el nuevo rostro a la lista de datos de rostros
                nuevo_rostro = {"nombre": nombre_de_la_imagen, "descriptor": descriptor_input}
                rostros_data.append(nuevo_rostro)
            else:
                return jsonify({"error": "No se detectaron rostros en al menos una de las imágenes."}), 400

        return jsonify({"mensaje": "Rostros insertados correctamente."})

    except Exception as e:
        print('Error al insertar el rostro:', str(e))
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error"}), 500


@app.errorhandler(Exception)
def handle_error(e):
    print(f"Error en la aplicación: {str(e)}")
    traceback.print_exc()
    return jsonify({"error": "Internal Server Error"}), 500


if __name__ == '__main__':
    cargar_datos_de_rostros()
    app.run(port=5000)
