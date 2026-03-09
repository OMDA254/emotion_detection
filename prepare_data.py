from utils import get_face_landmarks
import os   
import cv2
import numpy as np

data_dir = './data/test'
output  = []
for emotion_index,emotion in enumerate(sorted(os.listdir(data_dir))):
    emotion_dir = os.path.join(data_dir, emotion)
    for img_name in os.listdir(emotion_dir):
        img_path = os.path.join(emotion_dir, img_name)
        image = cv2.imread(img_path)
        landmarks = get_face_landmarks(image, draw=False, static_image_mode=True)
        if len(landmarks) == 1404:
            landmarks.append(int(emotion_index))
            output.append(landmarks)
np.savetxt('data.txt', np.asarray(output))



