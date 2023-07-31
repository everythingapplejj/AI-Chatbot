import cv2
import mediapipe as mp
from fer import FER
import random

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # 0 represents the default camera (front-facing camera)

# Initialize the Face detection detector
face_detector = FER(mtcnn=True)

# Initialize Mediapipe hands module
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Music suggestions based on emotion
music_suggestions = {
    'Angry': ['Song1', 'Song2', 'Song3'],
    'Happy': ['Song4', 'Song5', 'Song6'],
    'Sad': ['Song7', 'Song8', 'Song9'],
    'Neutral': ['Song10', 'Song11', 'Song12']
}

while True:
    # Read a frame from the video capture object
    success, img = cap.read()

    # Process hand tracking using Mediapipe
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Detect faces and emotions in the current frame
    frame_data = face_detector.detect_emotions(img)

    # Display the frame with emotion analysis
    for face in frame_data:
        x, y, w, h = face['box']
        emotion = max(face['emotions'], key=face['emotions'].get)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Music suggestion based on emotion
        if emotion in music_suggestions:
            songs = music_suggestions[emotion]
            suggested_song = random.choice(songs)
            cv2.putText(img, suggested_song, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()
