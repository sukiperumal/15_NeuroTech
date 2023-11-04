#Game of Narcos Breath rate detector
import cv2
import numpy as np
import time
cap = cv2.VideoCapture(0)
old_frame = None
starttimer = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)

    if old_frame is not None:
        frame_diff = cv2.absdiff(gray, old_frame)
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        elapse_time = time.time() - starttimer
        if elapse_time >= 5: 
            breath_rate = len(contours) / elapse_time
            print(f"Breath Rate: {breath_rate} breaths per second")
            starttimer = time.time()
        cv2.imshow("Breath Detection", frame)
    old_frame = gray
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

