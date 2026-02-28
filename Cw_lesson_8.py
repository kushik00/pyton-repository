import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('data/Haar_cascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/Haar_cascade/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('data/Haar_cascade/haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30)) #детектуємо обличчя

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) #перебираємо кожне обличчя і малюємо прямокутник

#roi - образ обличчя
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor= 1.1, minNeighbors= 10, minSize= (10,10)) #шукаємо очі саме в обличчі

        for (ex, ey, ew, eh) in eyes: #малюємо прямокутник
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (225, 0, 0), 2)

        smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(10, 10))

        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    cv2.putText(frame, f'Faces detected: {len(faces)}', (10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)


    cv2.imshow('Haar FAce Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
