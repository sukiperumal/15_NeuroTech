async def drowsiness_detection():
  # Load the necessary libraries
  import cv2
  import keras
  import numpy as np

  # Initialize the cascade classifiers for face detection and eye detection
  face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
  leye = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
  reye = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

  # Load the pre-trained Keras model for drowsiness detection
  model = keras.models.load_model('cnnCat2.h5')

  # Start a video capture loop
  cap = cv2.VideoCapture(0)

  # Keep track of the drowsiness score
  drowsiness_score = 0

  # Loop until the user presses the 'q' key
  while
 
True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and eyes in the grayscale frame
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    # For each detected face, predict the state of the eyes (open or closed) using the pre-trained Keras model
    for (x, y, w, h) in faces:
      # Get the left and right eye regions
      left_eye_region = gray[y:y + h, x:x + w // 2]
      right_eye_region = gray[y:y + h, x + w // 2:x + w]

      # Resize and normalize the eye regions
      left_eye_region = cv2.resize(left_eye_region, (24, 24)) / 255.0
      right_eye_region = cv2.resize(right_eye_region, (24, 24)) / 255.0

      # Predict the state of the eyes using the pre-trained Keras model
      left_eye_pred = model.predict(left_eye_region.reshape((1, 24, 24, 1)))
      right_eye_pred = model.predict(right_eye_region.reshape((1, 24, 24, 1)))

      # Update the drowsiness score
      drowsiness_score += (1 - left_eye_pred[0][0]) + (1 - right_eye_pred[0][0])

    # If the drowsiness score exceeds a certain threshold, play an alarm sound and display a warning message on the screen
    if drowsiness_score > 15:
      play_alarm_sound()
      display_warning_message()

    # Display the frame
    cv2.imshow('frame', frame)

    # If the user presses the 'q' key, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Release the camera
  cap.release()

  # Destroy all windows
  cv2.destroyAllWindows()

# Schedule the drowsiness detection coroutine using asyncio.ensure_future()
asyncio.ensure_future(drowsiness_detection())