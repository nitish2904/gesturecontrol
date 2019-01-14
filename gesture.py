
import cv2
import numpy as np
import math
from sound import Sound

cap = cv2.VideoCapture(0)

while (cap.isOpened()):

    ret, img = cap.read()
    cv2.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0)
    crop_img = img[100:300, 100:300]

    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 100)

    _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('Thresholded', thresh1)

    contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = max(contours, key=lambda x: cv2.contourArea(x))

    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 500), 0)

    hull = cv2.convexHull(cnt)

    drawing = np.zeros(crop_img.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

    hull = cv2.convexHull(cnt, returnPoints=False)

    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)
    i=0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])


        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57


        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img, far, 1, [255, 0, 0], 3)


        cv2.line(crop_img, start, end, [0, 255, 0], 2)

    if count_defects == 1:  # 2 fingers in the frame - 40% volume
        cv2.putText(img, "Volume 40%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        Sound.volume_up()
        continue
    elif count_defects == 2:  # 3 fingers in the frame - 60% volume
        cv2.putText(img, "Volume 60%", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        Sound.volume_down()
        continue
    elif count_defects == 3:  # 4 fingers in the frame - 80% volume
        cv2.putText(img, "Volume 80%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        Sound.volume_mute()
        continue
    elif count_defects == 4:  # 5 fingers in the frame - 100% volume
        cv2.putText(img, "Volume 100%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        Sound.volume_mute()
        continue

    cv2.imshow('Gesture', img)
    all_img = np.hstack((drawing, crop_img))
    cv2.imshow('Contours', all_img)
    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release();
cv2.destroyAllWindows()
