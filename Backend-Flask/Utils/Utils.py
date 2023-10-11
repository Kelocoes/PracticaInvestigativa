import PIL
import numpy as np
import cv2

def ImagePreprocessingFromRequest(data):
    img = PIL.Image.fromarray(cv2.cvtColor(data, cv2.COLOR_BGR2GRAY))
    input_shape = (1, 1, 64, 64)
    img = img.resize((64, 64), PIL.Image.LANCZOS)
    img_data = np.array(img)
    img_data = np.resize(img_data, input_shape)
    img_data = img_data.astype(np.float32)
    return img_data

def GetFirstFace(faces, image_np_array):
    source_face = faces[0]
    bbox = source_face["bbox"]
    bbox = [int(b) for b in bbox]
    face = image_np_array[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return face

def TransformEmotionResults(results):
    emotions = ["Neutral", "Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear", "Contempt"]
    results = list(zip(emotions, results))
    results.sort(key=lambda x: x[1], reverse=True)
    return {
        "Emotions": [results[0][0], results[1][0]],
        "Scores": [round(results[0][1],3), round(results[1][1],3)]
    }