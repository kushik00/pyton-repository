import cv2

image = cv2.imread('images/2.jpg')
image_resized = cv2.resize(image, (500, 600))
image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(image_gray, 100, 200)
cv2.imwrite('photo.jpg', edges)

cv2.imshow('photo', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


#


image = cv2.imread('images/3.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

cv2.imwrite('email.jpg', edges)
cv2.imshow('email', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
