import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
import numpy as np
import os

# Inicjalizacja MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

class MediapipeHandCrop:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.5, include_characteristic_vectors=False):
        self.include_characteristic_vectors = include_characteristic_vectors
        self.mp_hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )

    def __call__(self, image):
        # Check if image is a path
        if isinstance(image, str) and os.path.isfile(image):
            image = Image.open(image)
        # Obsługa obrazów typu PIL.Image
        if isinstance(image, Image.Image):
            image = np.array(image)  # Konwersja na numpy.ndarray
        # elif isinstance(image, np.ndarray):
            if image.shape[-1] == 3:  # Zakładamy obraz w formacie RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("Input should be a PIL.Image, numpy.ndarray, or a valid file path.")

        # Przetwarzanie obrazu za pomocą MediaPipe
        results = self.mp_hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the bounding box around the hand
                h, w, _ = image.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0

                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                # Calculate padding 
                total_pixels = h * w
                # padding = int(0.0001 * total_pixels)
                padding = int(min(h, w) * 0.03)  # 10% padding based on the smaller dimension

                x_min = max(x_min - padding, 0)
                y_min = max(y_min - padding, 0)
                x_max = min(x_max + padding, w)
                y_max = min(y_max + padding, h)

                # Ensure the cropped image is a square
                box_width = x_max - x_min
                box_height = y_max - y_min
                diff = abs(box_width - box_height)

                if box_width > box_height:
                    y_min = max(y_min - diff // 2, 0)
                    y_max = min(y_max + diff // 2, h)
                else:
                    x_min = max(x_min - diff // 2, 0)
                    x_max = min(x_max + diff // 2, w)

                # Crop the hand region
                cropped_hand = image[y_min:y_max, x_min:x_max]


            # Wyodrębnianie charakterystycznych wektorów, jeśli wymagane
            if self.include_characteristic_vectors:
                characteristic_vectors = np.array(
                    [[landmark.x, landmark.y] for landmark in hand_landmarks.landmark]
                )
                return cropped_hand[:, :, ::-1], characteristic_vectors
            


            return Image.fromarray(cropped_hand[:, :, ::-1])
        
        # Zwracanie None, jeśli nie wykryto dłoni
        return None
