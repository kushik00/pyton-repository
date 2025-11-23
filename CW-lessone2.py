import cv2
import numpy as np

image = cv2.imread('images/1.png')
print(image.shape)
# image = cv2.resize(image, (800, 500))
# image = cv2.resize(image, ( image.shape[1] // 2, image.shape[0] // 2))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image.shape)
image = cv2.Canny(image, 100, 100)


carnal = np.ones((5,5) , np.uint8)
# image = cv2.dilate(image , carnal , iteration=1) - розширення світлх зон
image = cv2.erode(image, carnal, iterations=1)
cv2.imwrite('1.jpg', image)

cv2.imshow('apple' , image)
cv2.waitKey(0)
cv2.destroyAllWindows()