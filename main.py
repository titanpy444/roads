import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(imageNew):
    image_gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    image_canny = cv2.Canny(image_blur, 70, 210)
    return image_canny


def createMask(imageNew):
    height = np.shape(imageNew)[0]
    mask = np.zeros_like(imageNew)
    triangle = np.array([[(650, 700), (1400, 730), (700, 350)]])
    cv2.fillPoly(mask, triangle, 255)
    image_crop = cv2.bitwise_and(imageNew, mask)
    return image_crop


def display_line(image_new, lins):
    line_image = np.zeros_like(image_new)
    if lins is not None:
        for line in lins:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 80, 0), 10)
    return line_image


image = cv2.imread('highway.jpg')
lane_image = np.copy(image)

# plt.imshow(canny(lane_image))
# plt.show()
can = canny(image)
cropped_image = createMask(can)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
image_line = display_line(lane_image, lines)
combo_image = cv2.addWeighted(lane_image, 1, image_line, 1, 1)
cv2.imshow('canny', combo_image)
cv2.waitKey(0)
