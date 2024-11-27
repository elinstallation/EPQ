#preprocessing
import cv2

def preprocess_face(face):
    #convert face to grayscale
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    #apply histogram equalization
    face_equalized = cv2.equalizeHist(face_gray)
    #resize the face to the required size (227x227)
    face_resized = cv2.resize(face, (227, 227))
    #denoise face
    face_denoised = cv2.fastNlMeansDenoising(face_resized, None, 30, 7, 21)
    return face_denoised
