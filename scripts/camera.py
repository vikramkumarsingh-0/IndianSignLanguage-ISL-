import cv2
from scripts.model import preprocess_frame

# Flag to stop the camera feed
camera_active = False

def start_camera_feed(model, labels):
    global camera_active
    # camera_active = True
    # cap = cv2.VideoCapture(0)

    # while camera_active:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

        # Preprocess the frame for prediction
        # processed_frame = preprocess_frame(frame)
        # predictions = model.predict(processed_frame)
        # predicted_label = labels[predictions.argmax()]

        # Display the predicted gesture
        # cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.imshow('Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def stop_camera_feed():
    global camera_active
    camera_active = False
