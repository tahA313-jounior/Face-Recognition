from deepface import DeepFace

def create_embedding(img_path):
    embedding = DeepFace.represent(
        img_path = img_path,
        detector_backend = 'retinaface',
        enforce_detection = False,
        model_name="ArcFace" 

    )
    return embedding[0]['embedding']