from flask import Flask, jsonify
import firebase_admin
from firebase_admin import credentials, storage, initialize_app
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


cred = credentials.Certificate( "D:/APIS/uniface-9cd0b-firebase-adminsdk-n0htw-229f9c1c57.json")
firebase_admin.initialize_app(cred, {"storageBucket": "uniface-9cd0b.appspot.com"})
bucket = storage.bucket()

@app.route('/', methods=['GET'])
def get_storage_data():
    try:
        blobs = bucket.list_blobs()
        storage_data = [{"name": blob.name, "link": f"https://firebasestorage.googleapis.com/v0/b/uniface-9cd0b.appspot.com/o/{blob.name}?alt=media&token={blob.public_url.split('=')[-1]}"} for blob in blobs if blob.name.lower().endswith(('.jpg', '.jpeg', '.png'))]
        return jsonify(storage_data)
    except Exception as e:
        print('Error obteniendo datos de Firebase Storage:', str(e))
        return jsonify({"error": "Internal Server Error"}), 500

def main(request):
        # Esta función se llama cuando se activa la función Cloud Function
        return app(request)

if __name__ == '__main__':
    app.run(port=8000)