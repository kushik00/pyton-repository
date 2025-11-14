import cv2
#
# img = cv2.imread('images/1.jpg')
# img = cv2.resize(img, (500,300))
#
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cap = cv2.VideoCapture('video/video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (400, 300))

    cv2.imshow('video', resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()