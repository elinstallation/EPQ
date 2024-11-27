import cv2
import torch  
from preprocessing import preprocess_face
from detection import detect_faces
from prediction import load_models, predict_age_gender
from logging_utils import log_with_timestamp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    age_net, gender_net = load_models(device)

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        face_locations = detect_faces(frame)

        for (top, right, bottom, left) in face_locations:
            face = frame[top:bottom, left:right]
            processed_face = preprocess_face(face)
            gender, age = predict_age_gender(processed_face, age_net, gender_net)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, f"{gender}, {age}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            log_with_timestamp(gender, age)

        cv2.imshow("Video", frame)

        #stop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
