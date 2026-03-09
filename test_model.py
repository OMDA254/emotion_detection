import cv2
import pickle

from utils import get_face_landmarks

model = pickle.load(open('emotion_model.pkl', 'rb'))

emotion_labels = ["happy", "sad", "angry"]

cap = cv2.VideoCapture(0)

ret, fame = cap.read()

while ret:
    ret, frame = cap.read()

    landmarks = get_face_landmarks(frame, draw=True, static_image_mode=False)

    output = model.predict([landmarks])


    cv2.putText(frame,
                emotion_labels[int(output[0])],
                 (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0), 2)


    cv2.imshow('frame', frame)
    
    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()