import cv2

cap = cv2.VideoCapture(0) # ID of my webcam

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret == False:
        continue
    
    cv2.imshow("Video Frame", frame)
    cv2.imshow("Gray Frame" ,gray_frame)

    # Wait for user input - q, then this will break the loop

    keyPressed = cv2.waitKey(1) & 0xFF # Masking cv2.waitkey(1), which is 32 bit to  bit number

    if keyPressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()