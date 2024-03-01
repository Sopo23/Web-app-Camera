import cv2
import numpy as np
from playsound import playsound

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Read the first frame
_, frame1 = cap.read()
_, frame2 = cap.read()

try:
    while cap.isOpened():
        # Calculate the absolute difference between frames
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Check for significant changes
        for contour in contours:
            if cv2.contourArea(contour) > 5000:
                playsound('alarm.mp3') # Make sure you have an alarm.mp3 file in your working directory
                break

        # Show the video feed
        cv2.imshow("Surveillance Camera", frame1)

        # Update the frames
        frame1 = frame2
        _, frame2 = cap.read()

        # Break the loop with the 'q' key
        if cv2.waitKey(40) == ord('q'):
            break

except Exception as e:
    print("An error occurred:", e)

finally:
    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()
