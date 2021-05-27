#Importing libraries
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# reading image
bottle = cv.imread("bot1.jpeg")
org = cv.resize(bottle, (471, 671))  # cloning original image to use in final result
bottle = cv.cvtColor(bottle, cv.COLOR_BGR2RGB)
bottle = cv.resize(bottle, (471, 671))

# cv.imshow("Bottle",bottle)
plt.imshow(bottle)
plt.title("Original image")
plt.show()

# printing size
'''
rows,column,d=bottle.shape
print("(H,W,D):" ,bottle.shape)
print("size:",(bottle.size))
print("pixels:",bottle)
'''
# Grayscale_image
red_frame = bottle[:, :, 0]
cv.imshow("red frame", red_frame)
cv.waitKey(0)

# drawing histogram
plt.hist(red_frame.ravel(), 256, [0, 256])
plt.title("Histogram of redframe")
plt.show()

# eliminating background
for row in range(red_frame.shape[0]):
    for col in range(red_frame.shape[1]):
        if red_frame[row, col] > 100:
            red_frame[row, col] = 1
cv.imshow("background masked", red_frame)
cv.waitKey()

for row in range(red_frame.shape[0]):
    for col in range(red_frame.shape[1]):
        if red_frame[row, col] < 100 and red_frame[row, col] != 1:
            red_frame[row, col] = 255
cv.imshow("liquid masked", red_frame)
cv.waitKey()

# smoothening operation
bottle_smooth = cv.GaussianBlur(red_frame, (7, 7), 0)
cv.imshow("Smooth bottle", bottle_smooth)
cv.waitKey(0)

# OTSU's Thresholding
#ret1, th1 = cv.threshold(red_frame, 0, 255, cv.THRESH_BINARY)
#ret2, th2 = cv.threshold(red_frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
ret3, th3 = cv.threshold(bottle_smooth, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow("thresholding", th3)
#cv.imshow("thresholding2", th2)
cv.waitKey()

# dialation operation
kernel = np.ones((9, 9), np.uint8)
dilation = cv.dilate(th3, kernel, iterations=1)
cv.imshow("dilation", dilation)
cv.waitKey(0)

# opening and closing operations
opening = cv.morphologyEx(dilation, cv.MORPH_OPEN, kernel)
cv.imshow("opening operation", opening)
cv.waitKey(0)
closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
cv.imshow("closing  operation", closing)
cv.waitKey(0)

# Erosion to reduce noise remaining
erosion = cv.erode(closing, kernel, iterations=3)
cv.imshow("erosion", erosion)
cv.waitKey(0)

# canny detection
canny = cv.Canny(erosion, 100, 150)
cv.imshow("canny", canny)
cv.waitKey(0)
cv.destroyAllWindows()

# Find bounding box
x, y, w, h = cv.boundingRect(canny)
print("height=", h)
cv.rectangle(org, (x, y), (x + w, y + h), (36, 255, 12), 2)
# cv.putText(org, "w={},h={}".format(w,h), (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)

# Drawing Levels
if h > 380:
    cv.putText(org, "Bottle is 100% Full", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.imshow("Full bottle", org)
    cv.waitKey(0)
elif h < 380 and h > 320:
    cv.putText(org, "Bottle is 90% Full", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.imshow("90% Full bottle", org)
    cv.waitKey(0)
elif h < 320 and h > 240:
    cv.putText(org, "Bottle is 75% Full", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.imshow("75% Full bottle", org)
    cv.waitKey(0)
elif h < 240 and h > 145:
    cv.putText(org, "Bottle is 50% Full", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.imshow("50% Full bottle", org)
    cv.waitKey(0)
else:
    cv.putText(org, "Bad Bottle", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.imshow("less than 30%", org)
    cv.waitKey(0)
# The end
cv.destroyAllWindows()
