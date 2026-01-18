import cv2
import numpy as np

img = cv2.imread("input.jpg")
result = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

clahe = cv2.createCLAHE(2.0, (8, 8))
gray = clahe.apply(gray)

edges = cv2.Canny(gray, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

count = 0

for c in contours:
    x, y, w, h = cv2.boundingRect(c)

    area = w * h
    ratio = max(w, h) / (min(w, h) + 1)

    if area > 40000 and ratio < 30000:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        count += 1
print("магнітів:", count)
cv2.imwrite("result.jpg", result)

