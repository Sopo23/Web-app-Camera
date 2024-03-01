from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

cap = cv2.VideoCapture(0)

def gen_frames():
    success, frame1 = cap.read()
    if not success:
        return  # Exit the function if the first frame is not captured

    while True:
        success, frame2 = cap.read()
        if not success:
            break  # Break the loop if a frame is not captured

        # Motion detection logic
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        movement_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 5000:
                movement_detected = True
                break
            # Convert frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame1)
        if not ret:
            break  # Break the loop if frame conversion fails

        frame1 = buffer.tobytes()

        # Yield frame in multipart/x-mixed-replace format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')

        # Update frame1 for the next loop iteration
        frame1 = frame2

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True)
