import cv2
import numpy as np
from matplotlib import pyplot as plt








path = "motherboard_image.jpeg"


image_real = cv2.imread(path, cv2.IMREAD_COLOR)
image_real = cv2.rotate(image_real, cv2.ROTATE_90_CLOCKWISE)

image = cv2.imread(path, cv2.IMREAD_COLOR)
image = cv2.GaussianBlur(image, (47, 47), 4)
image_pink = cv2.cvtColor(image, cv2.COLOR_RGB2PINK)
image_pink = cv2.adaptiveThreshold(image_pink, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55,7)
image_pink = cv2.rotate(image_pink, cv2.ROTATE_90_CLOCKWISE)




edges = cv2.Canny(image_pink, 50, 300)
edges = cv2.dilate(edges,None, iterations = 10)

contours, cow = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros_like(image_real)

cv2.drawContours(image=mask, contours=[max(contours, key = cv2.contourArea)], contourIdx=-1, color=(255, 255, 255), thickness= cv2.FILLED)

masked_image =cv2.bitwise_and(mask, image_real)

plt.imshow(masked_image)

plt.show()


