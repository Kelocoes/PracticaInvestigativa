from flask import Flask, jsonify, request
from flask_cors import CORS
import onnxruntime
import PIL
import base64
import numpy as np
import io
from insightface.app import FaceAnalysis
from Utils.Utils import ImagePreprocessingFromRequest, GetFirstFace, TransformEmotionResults

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

providers=["CUDAExecutionProvider","CPUExecutionProvider"]
face_recognition_model = "buffalo_s"
emotion_recognition_model_path = "../Models/emotion-ferplus-12-int8.onnx"

sesion = onnxruntime.InferenceSession(emotion_recognition_model_path, providers=providers)
face_recognition = FaceAnalysis(name=face_recognition_model, providers=providers)
face_recognition.prepare(ctx_id=0, det_size=(640, 640))

@app.route('/emotion-recognition', methods=['POST'])
def obtener_datos():
    try:
        json_data = request.get_json()
        image_base64 = json_data['image'].encode('utf-8')
        image_data = base64.b64decode(image_base64)
        image_np_array = np.array(PIL.Image.open(io.BytesIO(image_data)))

        faces = face_recognition.get(image_np_array)
        face = GetFirstFace(faces, image_np_array)
        img_data = ImagePreprocessingFromRequest(face)
        results = sesion.run(None, {"Input3": img_data})
        emotion_results = TransformEmotionResults(results[0][0].tolist())
        return jsonify({
                'InsightFaceResults': {
                    'score': round(float(faces[0]['det_score']),3),
                    'gender': "Woman" if faces[0]['gender'] == 0 else "Man" ,
                    'age': faces[0]['age']
                },
                'EmotionResults': emotion_results
            })

    except Exception as e:
        return jsonify({'error': 'Ha ocurrido un error al realizar el procesamiento',
                        "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)