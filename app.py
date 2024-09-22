from flask import Flask, render_template, Response, jsonify
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

app = Flask(__name__)

# Initialize the camera and hand detector
camera = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.8)  # Set detection confidence to 0.8 and reduce drawing
classifier = Classifier("models/keras_model.h5", "models/labels.txt")
offset = 20
imgSize = 300

labels = ["bad", "call me", "good", "heart", "hello", "i love you", "ok", "please", "thank you"]

# Global variable for detected gesture
detected_gesture = "None"

@app.route('/')
def index():
    return render_template('index.html')

def generate_video_feed():
    global detected_gesture
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Hand detection
        hands, frame = detector.findHands(frame, draw=False)  # Disable drawing of keypoints and text like 'Right'
        if hands:
            hand = hands[0]  # Get the first hand detected
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = frame[y-offset:y+h+offset, x-offset:x+w+offset]
            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal+hGap, :] = imgResize

            # Prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            detected_gesture = labels[index]

            # Draw output (without background and smaller text)
            cv2.putText(frame, labels[index], (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Smaller text
            cv2.rectangle(frame, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)  # Box around hand

        # Encode the frame as a JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    global camera
    if not camera.isOpened():
        camera.open(0)
    return "Camera started"

@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera.isOpened():
        camera.release()
    return "Camera stopped"

@app.route('/get_gesture')
def get_gesture():
    return jsonify({'gesture': detected_gesture})

if __name__ == '__main__':
    app.run(debug=True)
