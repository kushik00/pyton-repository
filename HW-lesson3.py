import cv2

img = cv2.imread("images/2.jpg")
cv2.rectangle(img, (480, 200), (360, 400), (0, 255, 0), 3)

cv2.putText(img, "Shcherbakova Lada", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


cv2.imshow("Result", img)

cv2.imwrite("HW_lesson_3_result.jpg", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
