import cv2

# Load the cascade classifiers
face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Create a region of interest (ROI) for the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # Detect eyes within the ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.5, 10)
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around the eyes
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()