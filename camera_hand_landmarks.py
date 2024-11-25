import cv2
import mediapipe as mp

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Webcam
camera = cv2.VideoCapture(0)

# Mediapipe Hands configuration
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to capture video")
            break

        # Convert frame to RGB (Mediapipe expects RGB input)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hand landmarks
        results = hands.process(frame_rgb)

        # If landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Extract and print landmark coordinates
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                print(landmarks)  # Debugging: Print landmarks to console

        # Display the frame with landmarks
        cv2.imshow("Hand Landmarks", frame)

        # Break loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release camera and close windows
camera.release()
cv2.destroyAllWindows()
