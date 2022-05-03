import cv2

img = cv2.imread('Images/me.jpg')
greyImg = cv2.imread('Images/me.jpg', cv2.IMREAD_GRAYSCALE)

# We're using imshow() of cv2 itself, so it would interprete RGB as RGB itself.
cv2.imshow("My Image", img)
cv2.imshow("Grey Image", greyImg)

# 0 means infinite, Number inside indicates milli seconds.
cv2.waitKey(2000)
cv2.destroyAllWindows()