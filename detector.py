from deepface import DeepFace
import supervision as sv
import numpy as np 
def detect_face(image):
    faces = DeepFace.extract_faces(
        img_path=image,
        detector_backend="retinaface",
            enforce_detection=False
    )
    boxes = []
    confidences = []
    for face in faces:
        facial_area = face['facial_area']
        x1 = facial_area['x']
        y1 = facial_area['y']
        x2 = facial_area['w'] + x1
        y2 = facial_area['y'] + y1
        confidence = face['confidence']
        boxes.append([x1, y1, x2, y2])
        confidences.append(confidence)
    detections = sv.Detections(
    xyxy=np.array(boxes),
    confidence=np.array(confidences)
)
    return(faces, detections, boxes)   