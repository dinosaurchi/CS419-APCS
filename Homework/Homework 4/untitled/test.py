__author__ = 'macpro'

import cv2

img = cv2.imread('2560-11.jpg', 0)
cv2.imshow('fuck', img)
cv2.waitKey(0)
cv2.destroyAllWindows()