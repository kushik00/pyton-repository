from os.path import split

import cv2
import numpy as np

osnow_image = cv2.imread('./photos/candyes.jpg')
new_width = 400
new_height = 300
image = cv2.resize(osnow_image, (new_width, new_height))

if image is not None:
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    yellow_lower = np.array([24, 0, 0])
    yellow_upper = np.array([92, 255, 255])

    pinky_lower = np.array([151, 44, 0])
    pinky_upper = np.array([179, 255, 255])

    purple_lower = np.array([99, 37, 0])
    purple_upper = np.array([135,204, 255])

    mask_yellow = cv2.inRange(image_hsv, yellow_lower, yellow_upper)
    mask_pinky = cv2.inRange(image_hsv, pinky_lower, pinky_upper)
    mask_purple = cv2.inRange(image_hsv, purple_lower, purple_upper)

    mask_final = cv2.bitwise_or(mask_yellow, cv2.bitwise_or(mask_pinky, mask_purple))
    result = cv2.bitwise_and(image, image, mask=mask_final)

    contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 600:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, ' ', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite('./photos/result.jpg', image)

    colour_first = cv2.putText(image, 'pink', (70, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    colour_secound = cv2.putText(image, 'yellow', (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    colour_last = cv2.putText(image, 'purple', (200, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("image hsv", result)
    cv2.imshow("image", image)

    cv2.waitKey(0)






