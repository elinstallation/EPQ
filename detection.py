from mtcnn import MTCNN #algorithm that uses to for face detection 
import cv2

detector = MTCNN()

def detect_faces(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)
    face_locations = []

    for face in faces:
        x, y, width, height = face['box'] # puts a box around face
        face_locations.append((y, x + width, y + height, x)) 
        
    return face_locations
